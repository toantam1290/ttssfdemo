import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor
import gradio as gr

# initialize fastspeech2 model.
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")


# initialize mb_melgan model
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")


# inference
processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

def inference(text):
  input_ids = processor.text_to_sequence(text)
  # fastspeech inference
  
  mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
      input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
      speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
      speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
      f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
      energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
  )

  # melgan inference
  audio_before = mb_melgan.inference(mel_before)[0, :, 0]
  audio_after = mb_melgan.inference(mel_after)[0, :, 0]
  
  # save to file
  sf.write('./audio_before.wav', audio_before, 22050, "PCM_16")
  sf.write('./audio_after.wav', audio_after, 22050, "FLOAT")
  return './audio_after.wav'
  
inputs = gr.inputs.Textbox(lines=10, label="Input Text")
outputs =  gr.outputs.Audio(type="file", label="Output Audio")


title = "Softfoundry Text to Speech Demo"
description = "To use it, please add  a short text in the input,  or click one of the examples to load them. then watch the audio file in the ouput,"

examples = [
  ["The English Wikipedia has an arbitration committee (also known as ArbCom) that consists of a panel of editors that imposes binding rulings with regard to disputes between other editors of the online encyclopedia."],
  ["Such users may seek information from the English Wikipedia rather than the Wikipedia of their native language because the English Wikipedia tends to contain more information about general subjects."],
  ["Bill got in the habit of asking himself “ Is that thought true ?” And if he wasn’t absolutely certain it was, he just let it go."]          
]
gr.Interface(inference, inputs, outputs, title=title, description=description, examples=examples).launch()
