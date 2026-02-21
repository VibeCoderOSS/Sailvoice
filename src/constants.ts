import type { ModelId } from './types/models';

export const MODELS: Array<{ id: ModelId; label: string; hfRepo: string }> = [
  {
    id: 'base',
    label: 'Base (8bit default)',
    hfRepo: 'mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit'
  },
  {
    id: 'customvoice',
    label: 'CustomVoice (8bit default)',
    hfRepo: 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit'
  },
  {
    id: 'voicedesign',
    label: 'VoiceDesign (8bit default)',
    hfRepo: 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit'
  }
];
