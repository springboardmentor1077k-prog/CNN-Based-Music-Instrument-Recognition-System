import numpy as np

CLASS_NAMES = [
    "Accordion",
    "Acoustic_Guitar",
    "Banjo",
    "Bass_Guitar",
    "Clarinet",
    "Cymbals",
    "Dobro",
    "Drum_set",
    "Electro_Guitar",
    "Floor_Tom",
    "Harmonica",
    "Harmonium",
    "Hi_Hats",
    "Horn",
    "Keyboard",
    "Mandolin",
    "Organ",
    "Piano",
    "Saxophone",
    "Shakers",
    "Tambourine",
    "Trombone",
    "Trumpet",
    "Ukulele",
    "Violin",
    "cowbell",
    "flute",
    "vibraphone"
]

np.save("features/class_names.npy", np.array(CLASS_NAMES))
print("✅ class_names.npy created")
