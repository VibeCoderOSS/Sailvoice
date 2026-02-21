from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .config import Settings
from .manager import JobManager
from .schemas import (
    JobCreatedResponse,
    LanguageDecisionRequest,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelInfo,
    JobRequestPdf,
    JobRequestText,
    RuntimeStatusResponse,
    VoiceCreateRequest,
    VoiceDesignCreateRequest,
    VoicePreviewRequest,
    VoicePreviewResponse,
    VoiceUpdateRequest,
)

settings = Settings.from_env()
manager = JobManager(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await manager.start()
    try:
        yield
    finally:
        await manager.stop()


app = FastAPI(title='Qwen3-TTS Local API', version='0.1.0', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
async def health() -> JSONResponse:
    runtime_status = await manager.get_runtime_status()
    return JSONResponse({'status': 'ok', 'backend': manager.engine.backend_label, 'runtime': runtime_status})


@app.get('/v1/jobs')
async def list_jobs() -> JSONResponse:
    jobs = await manager.list_jobs()
    return JSONResponse([job.model_dump(mode='json') for job in jobs])


@app.get('/v1/jobs/{job_id}')
async def get_job(job_id: str) -> JSONResponse:
    job = await manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return JSONResponse(job.model_dump(mode='json'))


@app.delete('/v1/jobs/{job_id}')
async def delete_job(job_id: str) -> JSONResponse:
    try:
        await manager.delete_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Job not found') from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({'ok': True})


@app.post('/v1/jobs/text', response_model=JobCreatedResponse)
async def create_text_job(request: JobRequestText) -> JobCreatedResponse:
    try:
        job_id = await manager.create_text_job(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JobCreatedResponse(id=job_id)


@app.post('/v1/jobs/pdf', response_model=JobCreatedResponse)
async def create_pdf_job(request: JobRequestPdf) -> JobCreatedResponse:
    try:
        job_id = await manager.create_pdf_job(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JobCreatedResponse(id=job_id)


@app.get('/v1/jobs/{job_id}/events')
async def stream_job_events(request: Request, job_id: str) -> StreamingResponse:
    job = await manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')

    async def event_generator():
        queue = await manager.subscribe(job_id)
        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ': keep-alive\n\n'
                    continue

                yield f'data: {json.dumps(event)}\n\n'
                if event.get('type') in {'done', 'error'}:
                    break
        finally:
            await manager.unsubscribe(job_id, queue)

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.get('/v1/assets/{asset_id}')
async def get_asset(asset_id: str) -> FileResponse:
    path = manager.get_asset_path(asset_id)
    if not path:
        raise HTTPException(status_code=404, detail='Asset not found')
    return FileResponse(path)


@app.get('/v1/voices')
async def list_voices() -> JSONResponse:
    voices = await manager.list_voices()
    return JSONResponse([voice.model_dump(mode='json') for voice in voices])


@app.post('/v1/voices')
async def create_voice(request: VoiceCreateRequest) -> JSONResponse:
    voice = await manager.create_voice(request)
    return JSONResponse(voice.model_dump(mode='json'))


@app.post('/v1/voices/design')
async def create_design_voice(request: VoiceDesignCreateRequest) -> JSONResponse:
    voice = await manager.create_design_voice(request)
    return JSONResponse(voice.model_dump(mode='json'))


@app.patch('/v1/voices/{voice_id}')
async def rename_voice(voice_id: str, request: VoiceUpdateRequest) -> JSONResponse:
    try:
        voice = await manager.rename_voice(voice_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Voice not found') from exc
    return JSONResponse(voice.model_dump(mode='json'))


@app.delete('/v1/voices/{voice_id}')
async def delete_voice(voice_id: str) -> JSONResponse:
    await manager.delete_voice(voice_id)
    return JSONResponse({'ok': True})


@app.post('/v1/voices/preview', response_model=VoicePreviewResponse)
async def preview_voice(request: VoicePreviewRequest) -> VoicePreviewResponse:
    try:
        asset_id = await manager.preview_voice(
            text=request.text,
            model_id=request.model,
            voice_id=request.voiceId,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return VoicePreviewResponse(assetId=asset_id)


@app.get('/v1/models', response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    payload = await manager.list_models()
    return [ModelInfo.model_validate(item) for item in payload]


@app.post('/v1/models/download', response_model=ModelDownloadResponse)
async def download_model(request: ModelDownloadRequest) -> ModelDownloadResponse:
    try:
        payload = await manager.download_model(model=request.model, source=request.source)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelDownloadResponse.model_validate(payload)


@app.get('/v1/runtime', response_model=RuntimeStatusResponse)
async def runtime_status() -> RuntimeStatusResponse:
    payload = await manager.get_runtime_status()
    return RuntimeStatusResponse.model_validate(payload)


@app.post('/v1/jobs/{job_id}/language')
async def confirm_job_language(job_id: str, request: LanguageDecisionRequest) -> JSONResponse:
    try:
        await manager.confirm_job_language(job_id=job_id, language=request.language)
    except KeyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({'ok': True})
