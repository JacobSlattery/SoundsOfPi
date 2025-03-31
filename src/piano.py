import numpy as np
import simpleaudio as sa
from mpmath import mp

from sounds import generate_sine_wave, apply_envelope, phase_align_wave

DIGIT_TO_KEY = {
    0: "C4", 1: "D4", 2: "E4", 3: "F4", 4: "G4",
    5: "A4", 6: "B4", 7: "C5", 8: "D5", 9: "E5"
}

# Generate a library of piano key frequencies
def create_piano_key_library():
    """
    Create a dictionary mapping piano key names to their frequencies.
    Returns:
        dict: A dictionary of piano keys and their corresponding frequencies.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    piano_keys = {}

    for i in range(88):
        # MIDI note number: Start at A0 = MIDI note 21
        midi_number = i + 21

        # Calculate the frequency for the key
        frequency = 440.0 * 2 ** ((midi_number - 69) / 12.0)  # MIDI 69 is A4

        # Determine the note name and octave
        note_index = midi_number % 12  # Position within the octave
        octave = (midi_number // 12) - 1  # Calculate octave (MIDI 12-23 is octave 0, etc.)
        note_name = f"{note_names[note_index]}{octave}"

        piano_keys[note_name] = round(frequency, 2) 

    return piano_keys

PIANO_KEYS = create_piano_key_library()


def generate_scale(root, scale_type="major", include_octaves=False):
    """
    Generates a scale based on the given root note and type.
    
    Args:
        root (str): Root note (e.g., "C4").
        scale_type (str): Type of scale ("major", "minor").
        include_octaves (bool): If True, include all octaves of the scale.
    
    Returns:
        list: List of note names in the scale.
    """
    major_steps = [2, 2, 1, 2, 2, 2, 1]
    minor_steps = [2, 1, 2, 2, 1, 2, 2]

    # Get the note names and find the index of the root
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_name, octave = root[:-1], int(root[-1])

    if root_name not in note_names:
        raise ValueError(f"Invalid root note: {root}")

    start_idx = note_names.index(root_name)
    scale_steps = major_steps if scale_type == "major" else minor_steps
    scale = [root]

    current_idx = start_idx
    for step in scale_steps:
        current_idx = (current_idx + step) % 12
        if current_idx < start_idx:
            octave += 1  # Move to next octave
        scale.append(f"{note_names[current_idx]}{octave}")

    if include_octaves:
        scale += [increase_octave(note) for note in scale]  # Add next octave

    return scale

# Define the function to play a single note or a chord
def play_notes(notes, duration=1, volume=0.5, sample_rate=44100):
    """
    Plays a single note or a combination of notes (chord).
    Args:
        notes (list or str): A piano key name (e.g., 'C4') or a list of key names.
        duration (float): Duration of the note in seconds.
        volume (float): Volume of the sound (0.0 to 1.0).
        sample_rate (int): Sampling rate for audio playback.
    """
    # Convert note names to frequencies
    if not isinstance(notes, list):
        notes = [notes]

    frequencies = [PIANO_KEYS[note] for note in notes if note in PIANO_KEYS]

    if not frequencies:
        raise ValueError("No valid notes provided or unrecognized note names.")

    # Generate sine waves for each frequency and combine them
    wave = sum(generate_sine_wave(freq, duration, sample_rate, amplitude=volume) for freq in frequencies)

    # Normalize the wave to fit in the range -1.0 to 1.0
    wave = wave / np.max(np.abs(wave))

    # Convert to 16-bit PCM audio format
    audio = (wave * 32767).astype(np.int16)

    # Play the audio using simpleaudio
    play_obj = sa.play_buffer(audio, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
    play_obj.wait_done()  # Wait until the audio finishes playing


def play_pi_sequence(digits=100, duration=0.5):
    """
    Plays the first 'digits' of π, mapping each digit (0–9) to a piano key.
    Args:
        digits (int): Number of π digits to play.
        duration (float): Duration of each note.
    """
    # Set π precision
    mp.dps = digits + 2  # Digits + "3."
    pi_digits = str(mp.pi)[2:]  # Remove "3."

    print(f"Playing the first {digits} digits of π as notes...")

    # Play each digit as a note
    for digit in pi_digits[:digits]:
        key = DIGIT_TO_KEY[int(digit)]
        print(f"Digit {digit} -> Key {key}")
        play_notes(key, duration=duration)

def play_pi_sequence_continuous(digits=100, duration=0.5, crossfade=0.1):
    """
    Plays the first 'digits' of π as a continuous sequence of notes with smooth, phase-aligned transitions.
    Args:
        digits (int): Number of π digits to play.
        duration (float): Duration of each note.
        crossfade (float): Overlapping duration between consecutive notes (for smooth transition).
    """
    mp.dps = digits + 2  # Digits + "3."
    pi_digits = str(mp.pi)[2:]  # Remove "3."

    print(f"Playing the first {digits} digits of π as continuous notes...")

    combined_wave = []  # Use a Python list for waveform collection
    sample_rate = 44100
    previous_wave = None  # Store previous wave for crossfading

    for i, digit in enumerate(pi_digits[:digits]):
        key = DIGIT_TO_KEY.get(int(digit), None)
        if key is None or key not in PIANO_KEYS:
            print(f"Digit {digit} -> Key not found in mapping!")
            continue

        print(f"Digit {digit} -> Key {key}")

        # Generate sine wave
        wave = generate_sine_wave(PIANO_KEYS[key], duration, sample_rate)
        print(f"Generated wave for {key} with length {len(wave)}")

        # Apply smooth attack and decay envelope
        wave = apply_envelope(wave, attack=0.02, decay=0.02)
        print(f"Applied envelope to wave for {key}")

        # Phase-align wave to ensure it starts and ends near zero-crossing
        wave = phase_align_wave(wave, sample_rate)
        print(f"Phase-aligned wave for {key}")

        if previous_wave is not None:
            crossfade_samples = min(len(previous_wave), int(sample_rate * crossfade))

            if crossfade_samples > 0:
                sine_fade = np.sin(np.linspace(0, np.pi / 2, crossfade_samples)) ** 2

                # Blend previous wave end with new wave start
                previous_wave[-crossfade_samples:] = (
                    previous_wave[-crossfade_samples:] * (1 - sine_fade) +
                    wave[:crossfade_samples] * sine_fade
                )
                wave = wave[crossfade_samples:]  # Remove blended section
                print(f"Crossfaded last {crossfade_samples} samples of {key}")

            # Append blended previous wave
            combined_wave.append(previous_wave)

        # Store current wave as previous for the next iteration
        previous_wave = wave

    # Append the final wave
    if previous_wave is not None:
        combined_wave.append(previous_wave)

    # Ensure we have valid waveform data
    if not combined_wave:
        print("No valid waveform generated. Exiting.")
        return

    # Convert list to a NumPy array
    combined_wave = np.concatenate(combined_wave)
    print(f"Final combined wave length: {len(combined_wave)}")

    # Normalize waveform to avoid clipping
    max_amplitude = np.max(np.abs(combined_wave))
    if max_amplitude > 0:
        combined_wave = combined_wave / max_amplitude
    print(f"Waveform normalized. Final length: {len(combined_wave)}")

    # Convert to 16-bit PCM audio format
    audio = (combined_wave * 32767).astype(np.int16)

    # Play the final waveform
    print("Playing audio...")
    play_obj = sa.play_buffer(audio, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
    play_obj.wait_done()

def get_harmonized_note(melody_note, scale_notes, harmony_type="third"):
    """
    Returns a harmonized note based on the melody note and scale.
    
    Args:
        melody_note (str): The melody note being played.
        scale_notes (list): List of notes in the scale.
        harmony_type (str): Type of harmony ("third", "fifth", "sixth").
    
    Returns:
        str: The best harmony note.
    """
    try:
        idx = scale_notes.index(melody_note)  # Find the melody note in the scale
    except ValueError:
        print(f"⚠️ Warning: {melody_note} not in scale! Using root.")
        return scale_notes[0]  # Fallback to root note

    harmony_intervals = {
        "third": 2,
        "fifth": 4,
        "sixth": 5
    }

    interval = harmony_intervals.get(harmony_type, 2)  # Default to third
    harmony_idx = (idx + interval) % len(scale_notes)  # Loop within the scale

    return scale_notes[harmony_idx]

def increase_octave(note):
    """
    Increases a note by one octave.
    
    Args:
        note (str): The note to shift up.
    
    Returns:
        str: The same note one octave higher.
    """
    if len(note) < 2 or not note[-1].isdigit():
        return note  # Return unchanged if invalid format

    note_name = note[:-1]  # Extract "C", "D#", etc.
    octave = int(note[-1]) + 1  # Increase octave

    return f"{note_name}{octave}"  # Return shifted note

def fix_wave_length(wave, target_length):
    """
    Adjusts a waveform to match the target length.
    
    Args:
        wave (numpy.array): Input waveform.
        target_length (int): The length to match.
    
    Returns:
        numpy.array: The resized waveform.
    """
    if len(wave) > target_length:
        return wave[:target_length]  # Trim if too long
    elif len(wave) < target_length:
        return np.pad(wave, (0, target_length - len(wave)), mode='constant')  # Pad with silence if too short
    return wave  # Already the correct length

def play_pi_sequence_with_harmony(
    digits=100, duration=0.5, crossfade=0.05, 
    key_root="C4", harmony_type="third",
    harmony_speed=2, octave_doubling=False,
    harmony_movement ="random"
):
    """
    Plays the first 'digits' of π as a melody while harmonizing it with pleasant intervals.
    
    Args:
        digits (int): Number of π digits to play.
        duration (float): Duration of each melody note.
        crossfade (float): Overlapping duration between consecutive notes.
        key_root (str): Root note for harmony selection.
        harmony_type (str): Type of harmony ("third", "fifth", "sixth").
        harmony_speed (int): Speed multiplier for harmony notes (e.g., 2 = twice as fast).
        octave_doubling (bool): Whether to play the harmony note an octave up as well.
        harmony_movement (str): How the harmony moves ("random", "intervals", "chordal").
    """
    mp.dps = digits + 2  # Digits + "3."
    pi_digits = str(mp.pi)[2:]  # Remove "3."

    print(f"Playing the first {digits} digits of π with harmonized accompaniment in {key_root}...")

    combined_wave = []  # Store waveforms
    sample_rate = 44100
    previous_wave = None  # Store previous wave for crossfading

    # Get a scale based on the key
    scale_notes = generate_scale(key_root, "major", include_octaves=True)  

    melody_duration = duration  
    harmony_duration = melody_duration / harmony_speed  

    harmony_counter = 0  # Track scale position for rolling harmony

    for i, digit in enumerate(pi_digits[:digits]):
        key = DIGIT_TO_KEY.get(int(digit), None)
        if key is None or key not in PIANO_KEYS:
            print(f"Digit {digit} -> Key not found in mapping!")
            continue

        print(f"🎵 Melody -> Playing {key}")

        # Generate π melody note
        melody_wave = generate_sine_wave(PIANO_KEYS[key], melody_duration, sample_rate)
        melody_wave = apply_envelope(melody_wave, attack=0.02, decay=0.02)
        melody_wave = phase_align_wave(melody_wave, sample_rate)

        # 🎼 Generate harmony notes correctly
        harmony_waves = []
        sustained_melody_wave = np.copy(melody_wave)  

        # Harmony Sequence Now Moves More Variably
        harmony_sequence = np.zeros_like(sustained_melody_wave)

        for h in range(harmony_speed):  
            # Harmony movement mode
            if harmony_movement == "random":
                scale_index = np.random.randint(0, len(scale_notes))  # Fully random choice
            elif harmony_movement == "intervals":
                scale_index = (harmony_counter + (h * 2)) % len(scale_notes)  # Jumping by intervals
            elif harmony_movement == "chordal":
                scale_index = (harmony_counter + (h * 3)) % len(scale_notes)  # Staggered chord-like movement
            else:
                scale_index = (harmony_counter + h) % len(scale_notes)  # Default fallback

            harmony_note = scale_notes[scale_index]

            print(f"🎹 Harmony -> Playing {harmony_note} at {harmony_duration}s duration (Position {h})")

            harmony_wave = generate_sine_wave(PIANO_KEYS[harmony_note], harmony_duration, sample_rate)
            harmony_wave = apply_envelope(harmony_wave, attack=0.02, decay=0.02)
            harmony_wave = phase_align_wave(harmony_wave, sample_rate)

            # Octave doubling volume is now balanced
            if octave_doubling:
                octave_up_note = increase_octave(harmony_note)
                if octave_up_note in PIANO_KEYS:
                    print(f"🎶 Octave Doubling -> Adding {octave_up_note}")
                    octave_wave = generate_sine_wave(PIANO_KEYS[octave_up_note], harmony_duration, sample_rate)
                    octave_wave = apply_envelope(octave_wave, attack=0.02, decay=0.02)
                    octave_wave = phase_align_wave(octave_wave, sample_rate)

                    # Lower volume of octave doubling to avoid overpowering
                    octave_wave *= 0.6  
                    harmony_wave *= 0.8  

                    harmony_waves.append(octave_wave)

            # Reduce harmony volume slightly to keep melody dominant
            harmony_wave *= 0.8  

            # Space harmony out sequentially
            start_idx = int(h * len(harmony_sequence) / harmony_speed)
            end_idx = start_idx + len(harmony_wave)

            if end_idx > len(harmony_sequence):
                end_idx = len(harmony_sequence)

            harmony_sequence[start_idx:end_idx] += harmony_wave[: (end_idx - start_idx)]

        # Move the harmony scale forward so the next melody starts at the next note
        harmony_counter = (harmony_counter + harmony_speed) % len(scale_notes)

        # Blend sustained melody into harmony
        combined = (sustained_melody_wave + harmony_sequence) / 2  

        if previous_wave is not None:
            crossfade_samples = min(len(previous_wave), int(sample_rate * crossfade))

            if crossfade_samples > 0:
                sine_fade = np.sin(np.linspace(0, np.pi / 2, crossfade_samples)) ** 2
                previous_wave[-crossfade_samples:] = (
                    previous_wave[-crossfade_samples:] * (1 - sine_fade) +
                    combined[:crossfade_samples] * sine_fade
                )
                combined = combined[crossfade_samples:]

            combined_wave.append(previous_wave)

        previous_wave = combined  

    if previous_wave is not None:
        combined_wave.append(previous_wave)

    if not combined_wave:
        print("No valid waveform generated. Exiting.")
        return

    combined_wave = np.concatenate(combined_wave)
    print(f"Final combined wave length: {len(combined_wave)}")

    # Normalize waveform
    max_amplitude = np.max(np.abs(combined_wave))
    if max_amplitude > 0:
        combined_wave = combined_wave / max_amplitude

    print("Playing audio...")
    audio = (combined_wave * 32767).astype(np.int16)
    play_obj = sa.play_buffer(audio, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
    play_obj.wait_done()