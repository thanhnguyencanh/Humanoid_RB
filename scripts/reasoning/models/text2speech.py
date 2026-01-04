import soundfile as sf
from kokoro import KPipeline
from IPython.display import display, Audio

# 1. Initialize the Pipeline
# lang_code='a' for American English, 'b' for British English
pipeline = KPipeline(lang_code='a')

# 2. Define your text
text = '''
Kokoro is an open-weight TTS model with 82 million parameters.
It is lightweight, fast, and sounds incredibly natural.
'''

# 3. Generate Audio
# voice='af_heart' is the default voice (50% Bella + 50% Sarah)
generator = pipeline(text, voice='af_heart', speed=1)

# 4. Save and Play
for i, (gs, ps, audio) in enumerate(generator):
    print(f"Generating segment {i}...")
    # Save to .wav file
    sf.write(f'output_{i}.wav', audio, 24000)
    print(f"Saved output_{i}.wav")