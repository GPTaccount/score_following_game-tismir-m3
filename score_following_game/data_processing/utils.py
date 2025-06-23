import glob
# - : librosa 模組相關功能已由自訂函式替代，故移除此匯入
# import librosa
import os
import tempfile
import time
import tqdm
import yaml

import numpy as np
import pretty_midi as pm

from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, \
    LogarithmicFilterbank


def load_game_config(config_file: str) -> dict:
    """Load game config from YAML file."""
    with open(config_file, 'rb') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return config


def merge_onsets(cur_onsets, stk_note_coords, coords2onsets):
    """ Merge onsets occurring in the same frame """

    # get coordinate keys
    coord_ids = coords2onsets.keys()

    # init list of unique onsets and coordinates
    onsets, coords = [], []

    # iterate coordinates
    for i in coord_ids:
        # check if onset already exists in list
        if cur_onsets[coords2onsets[i]] not in onsets:
            coords.append(stk_note_coords[i])
            onsets.append(cur_onsets[coords2onsets[i]])

    # convert to arrays
    coords = np.asarray(coords, dtype=np.float32)
    # - : np.int 已棄用，應改用更明確的 NumPy 整數型別
    # onsets = np.asarray(onsets, dtype=np.int)
    onsets = np.asarray(onsets, dtype=np.int32)
    # + : 將 np.int 替換為 np.int32，以符合 NumPy API 更新，並提供固定大小的整數型別

    return onsets, coords


def pad_representation(rep: np.ndarray, onset: np.ndarray, pad_offset: int, pad_value=0) -> (np.ndarray, np.ndarray):
    """Pad representation with zeros at the beginning and the end."""

    rep_out = None

    if rep.ndim == 2:
        rep_out = np.pad(rep, ((0, 0), (pad_offset, pad_offset)), mode='constant', constant_values=pad_value)

    if rep.ndim == 3:
        rep_out = np.pad(rep, ((0, 0), (0, 0), (pad_offset, pad_offset)), mode='constant', constant_values=pad_value)

    onset_out = onset + pad_offset

    return rep_out, onset_out


def load_data_from_dir(score_folder='score', perf_folder='performance', directory='test_sample',
                       real_perf=False):

    data = {}

    for cur_path_score in tqdm.tqdm(glob.glob(os.path.join(directory, score_folder, '*.npz'))):

        cur_path_perf = os.path.join(directory, perf_folder, os.path.split(cur_path_score)[-1])

        song_name = os.path.splitext(os.path.basename(os.path.normpath(cur_path_score)))[0]

        npzfile = np.load(cur_path_score, allow_pickle=True)
        score = (npzfile["sheet"], npzfile["coords"], npzfile['coord2onset'][0])

        if real_perf:
            perf = cur_path_perf.replace('.mid', '.wav')
        else:
            cur_path_perf = cur_path_perf.replace('.npz', '.mid')
            perf = pm.PrettyMIDI(cur_path_perf)

        data[song_name] = {'perf': perf, 'score': score}

    return data


def midi_reset_instrument(midi, id=0):
    """Set all instruments to `id`."""
    for cur_instr in midi.instruments:
        cur_instr.program = id
    return midi


def midi_reset_start(midi):
    """First onset is at position 0.
       Everything is adjusted accordingly.
    """
    for cur_instr in midi.instruments:
        # find index of first note, the first in the array is not necessarily the one with the earliest starting time
        first_idx = np.argmin([n.start for n in cur_instr.notes])
        first_onset = cur_instr.notes[first_idx].start
        for cur_note in cur_instr.notes:
            cur_note.start = max(0, cur_note.start - first_onset)

    return midi


def spectrogram_processor(spec_params):

    """Helper function for our spectrogram extraction."""
    sig_proc = SignalProcessor(num_channels=1, sample_rate=spec_params['sample_rate'])
    fsig_proc = FramedSignalProcessor(frame_size=spec_params['frame_size'], fps=spec_params['fps'])

    spec_proc = FilteredSpectrogramProcessor(filterbank=LogarithmicFilterbank, num_bands=12, fmin=60, fmax=6000,
                                             norm_filters=True, unique_filters=False)
    log_proc = LogarithmicSpectrogramProcessor()

    processor = SequentialProcessor([sig_proc, fsig_proc, spec_proc, log_proc])

    return processor


# + : 定義自訂頻譜正規化函式，用以取代 librosa.util.normalize。
#     此函式旨在模擬 librosa.util.normalize(norm=2, axis=0, threshold=0.01, fill=False) 的特定行為。
def _custom_normalize_spectrogram(spec_data: np.ndarray, norm_ord=2, axis=0, threshold=0.01) -> np.ndarray:
    """
    Custom normalization function to replace librosa.util.normalize with specific parameters.
    Normalizes columns of a spectrogram (axis=0).
    If a column's L2 norm (norm_ord=2) is below 'threshold', it's not modified,
    replicating fill=False behavior.
    """
    if axis != 0:
    # + : 檢查 axis 參數，目前此自訂函式僅支援 axis=0（按欄正規化），符合原始碼使用情境。
        raise NotImplementedError(f"axis={axis} is not implemented. Only axis=0 is supported for this custom normalizer.")

    spec_out = spec_data.copy()
    # + : 建立輸入頻譜的副本，以避免修改原始數據。

    for i in range(spec_data.shape[1]):  # spec_data.shape[1] is the number of frames (columns)
    # + : 遍歷頻譜中的每一欄（時間幀）。
        column = spec_data[:, i]
        col_norm = np.linalg.norm(column, ord=norm_ord)
        # + : 計算目前欄位的 L2-norm。

        if col_norm >= threshold:
        # + : 判斷欄位的 norm 是否達到正規化閾值。
            spec_out[:, i] = column / col_norm
            # + : 若達到閾值，則對該欄位進行正規化。
        # + : 若欄位 norm 未達到閾值，則根據 fill=False 的行為（即不填充0且不修改），該欄位在 spec_out 中保持不變。
        #     (This comments the implicit else logic for fill=False behavior)

    return spec_out
# + : 返回正規化處理後的頻譜數據。


def wav_to_spec(path_audio: str, spec_params: dict):
    """Extract spectrogram from audio."""

    processor = spectrogram_processor(spec_params)

    spec = processor.process(path_audio).T

    if spec_params.get('norm', False):
        # - : 根據重構任務指示，替換 librosa.util.normalize。
        # spec = librosa.util.normalize(spec, norm=2, axis=0, threshold=0.01, fill=False)
        spec = _custom_normalize_spectrogram(spec, norm_ord=2, axis=0, threshold=spec_params.get('norm_threshold', 0.01))
        # + : 使用自訂的 _custom_normalize_spectrogram 函式進行頻譜正規化，
        #     模擬原 librosa.util.normalize(norm=2, axis=0, threshold=0.01, fill=False) 的行為。
        #     允許通過 spec_params['norm_threshold'] 控制閾值。

    return np.expand_dims(spec, 0)


def midi_to_spec_otf(midi: pm.PrettyMIDI, spec_params: dict, sound_font_path=None) -> np.ndarray:
    """MIDI to Spectrogram (on the fly)

       Synthesizes a MIDI with fluidsynth and extracts a spectrogram.
       The spectrogram is directly returned
    """
    processor = spectrogram_processor(spec_params)

    def render_audio(midi_file_path, sound_font):
        """
        Render midi to audio
        """

        # split file name and extention
        name, extention = midi_file_path.rsplit(".", 1)

        # set file names
        audio_file = name + ".wav"

        # audio_file = tempfile.TemporaryFile('w+b')

        # synthesize midi file to audio
        cmd = "fluidsynth -F %s -O s16 -T wav %s %s 1> /dev/null" % (audio_file, sound_font, midi_file_path)

        os.system(cmd)
        return audio_file

    mid_path = os.path.join(tempfile.gettempdir(), str(time.time())+'.mid')

    with open(mid_path, 'wb') as f:
        midi.write(f)

    audio_path = render_audio(mid_path, sound_font=sound_font_path)

    spec = processor.process(audio_path).T

    if spec_params.get('norm', False):
        # - : 根據重構任務指示，替換 librosa.util.normalize。
        # spec = librosa.util.normalize(spec, norm=2, axis=0, threshold=0.01, fill=False)
        spec = _custom_normalize_spectrogram(spec, norm_ord=2, axis=0, threshold=spec_params.get('norm_threshold', 0.01))
        # + : 使用自訂的 _custom_normalize_spectrogram 函式進行頻譜正規化，
        #     模擬原 librosa.util.normalize(norm=2, axis=0, threshold=0.01, fill=False) 的行為。
        #     允許通過 spec_params['norm_threshold'] 控制閾值。


    # compute spectrogram
    spec = np.expand_dims(spec, 0)

    os.remove(mid_path)
    os.remove(audio_path)

    return spec


def midi_to_onsets(midi_file: pm.PrettyMIDI, fps: int, instrument_idx=None, unique=True) -> np.ndarray:
    """Extract onsets from a list of midi files. Only returns unique onsets."""

    if instrument_idx is not None:
        # only get onsets from the right hand
        onsets_list = (midi_file.instruments[instrument_idx].get_onsets()*fps).astype(int)
    else:
        # get all unique onsets merged in one list
        onsets_list = (midi_file.get_onsets()*fps).astype(int)

    return np.unique(onsets_list) if unique else onsets_list


def fluidsynth(midi, fs=44100, sf2_path=None):
    """Synthesize using fluidsynth.
    Copied and adapted from `pretty_midi`.

    Parameters
    ----------
    midi : pm.PrettyMidi
    fs : int
        Sampling rate to synthesize at.
    sf2_path : str
        Path to a .sf2 file.
        Default ``None``, which uses the TimGM6mb.sf2 file included with
        ``pretty_midi``.
    Returns
    -------
    synthesized : np.ndarray
        Waveform of the MIDI data, synthesized at ``fs``.
    """
    import os
    # ✅ 強制將 sf2_path 轉為絕對路徑（如果不是）
    if sf2_path is not None and not os.path.isabs(sf2_path):
        sf2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", sf2_path))

    # If there are no instruments, or all instruments have no notes, return
    # an empty array
    if len(midi.instruments) == 0 or all(len(i.notes) == 0 for i in midi.instruments):
        return np.array([])
    # Get synthesized waveform for each instrument
    waveforms = []
    for i in midi.instruments:
        if len(i.notes) > 0:
            print(f"[DEBUG] Trying to load sf2_path: {sf2_path}")
            waveforms.append(i.fluidsynth(fs=fs, sf2_path=sf2_path))

    # Allocate output waveform, with #sample = max length of all waveforms
    synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))

    # Sum all waveforms in
    for waveform in waveforms:
        synthesized[:waveform.shape[0]] += waveform

    # Scale to [-1, 1]
    synthesized /= 2**16
    # synthesized = synthesized.astype(np.int16)

    # normalize
    synthesized /= float(np.max(np.abs(synthesized)))

    return synthesized