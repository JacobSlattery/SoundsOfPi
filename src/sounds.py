import numpy as np

def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


def apply_envelope(wave, attack=0.02, decay=0.02, sample_rate=44100):
    """
    Apply an ADSR-like envelope with an attack and decay phase using a Hann window.
    """
    num_samples = len(wave)
    attack_samples = min(int(sample_rate * attack), num_samples // 2)
    decay_samples = min(int(sample_rate * decay), num_samples // 2)

    # Use Hann windows for attack and decay
    if attack_samples > 0:
        attack_curve = np.hanning(attack_samples * 2)[:attack_samples]  # First half of Hann window
        wave[:attack_samples] *= attack_curve

    if decay_samples > 0:
        decay_curve = np.hanning(decay_samples * 2)[-decay_samples:]  # Second half of Hann window
        wave[-decay_samples:] *= decay_curve

    return wave

def phase_align_wave(wave, sample_rate=44100):
    """
    Adjusts a waveform so it always starts and ends at a zero-crossing point.
    Prevents cutting the entire wave to an empty array.
    """
    # Find the closest zero-crossing at the start
    zero_crossings = np.where(np.diff(np.sign(wave)))[0]

    if len(zero_crossings) > 0:
        start_index = zero_crossings[0]
        end_index = zero_crossings[-1] if zero_crossings[-1] > start_index else len(wave) - 1

        # Ensure we don't trim the entire waveform
        if end_index - start_index > 0.1 * len(wave):  # Keep at least 10% of the wave
            wave = wave[start_index:end_index]
        else:
            print("Warning: phase_align_wave found too few zero-crossings! Keeping original wave.")

    return wave