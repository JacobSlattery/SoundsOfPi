import piano


# Example usage
piano.play_pi_sequence_with_harmony(digits=50, duration=1, crossfade=0.01, key_root = "C4", harmony_type="third", harmony_speed=4, octave_doubling=True, harmony_movement="chordal")

# Example usage
# Play a single note (A4)
# play_notes('A2', duration=1)

# # Play a chord (C4, E4, G4 = C major)
# play_notes(['C2', 'E2', 'G2'], duration=2)

# # Play a more complex chord
# play_notes(['A2', 'C#3', 'E3'], duration=2)