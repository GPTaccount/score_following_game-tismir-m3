import os
import sys
import types
import zipfile
import numpy as np
import pretty_midi as pm
from unittest.mock import patch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

# stub madmom modules required by utils
dummy = types.SimpleNamespace()
sys.modules.setdefault('madmom', types.ModuleType('madmom'))
sys.modules['madmom.processors'] = types.ModuleType('processors')
sys.modules['madmom.processors'].SequentialProcessor = object
sys.modules['madmom.audio'] = types.ModuleType('audio')
sys.modules['madmom.audio.signal'] = types.ModuleType('signal')
sys.modules['madmom.audio.signal'].SignalProcessor = object
sys.modules['madmom.audio.signal'].FramedSignalProcessor = object
sys.modules['madmom.audio.spectrogram'] = types.ModuleType('spectrogram')
sys.modules['madmom.audio.spectrogram'].FilteredSpectrogramProcessor = object
sys.modules['madmom.audio.spectrogram'].LogarithmicSpectrogramProcessor = object
sys.modules['madmom.audio.spectrogram'].LogarithmicFilterbank = object

from score_following_game.data_processing.utils import fluidsynth


def _make_dummy_midi():
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(0)
    inst.notes.append(pm.Note(velocity=100, pitch=60, start=0, end=1))
    midi.instruments.append(inst)
    return midi


def test_extract_zip_when_sf2_missing(tmp_path):
    midi = _make_dummy_midi()
    sf2 = tmp_path / "dummy.sf2"
    zip_path = sf2.with_suffix(sf2.suffix + ".zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dummy.sf2", "content")
    with patch.object(pm.Instrument, "fluidsynth", return_value=np.zeros(10)):
        fluidsynth(midi, sf2_path=str(sf2))
    assert sf2.exists(), "sound font should be extracted automatically"


def test_raise_error_when_sf2_not_found(tmp_path):
    midi = _make_dummy_midi()
    sf2 = tmp_path / "missing.sf2"
    with patch.object(pm.Instrument, "fluidsynth", return_value=np.zeros(10)):
        try:
            fluidsynth(midi, sf2_path=str(sf2))
        except RuntimeError:
            pass
        else:
            raise AssertionError("RuntimeError not raised for missing sound font")

