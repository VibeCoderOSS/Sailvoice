import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description='Qwen3-TTS local backend service')
    parser.add_argument('--port', type=int, default=int(os.getenv('TTS_PORT', '8765')))
    args = parser.parse_args()

    uvicorn.run('app.main:app', host='127.0.0.1', port=args.port, reload=False, log_level='info')


if __name__ == '__main__':
    main()
