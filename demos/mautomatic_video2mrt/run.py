# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Added by Iris Xu at 20200206, for personal usage.
import argparse
import codecs
import os
from typing import Optional

import numpy
import paddle
import soundfile
import librosa

from paddlespeech.cli import ASRExecutor
from paddlespeech.cli import TextExecutor
from paddlespeech.s2t.transform.transformation import Transformation

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--input", type=str, required=True)
args = parser.parse_args()


# yapf: enable

class MAutomaticVideo2MRTProducer:
    def __init__(self, asr_model_name: str = 'transformer_librispeech',
                 sample_rate: int = 16000,
                 asr_config: os.PathLike = None,
                 asr_ckpt_path: os.PathLike = None,
                 decode_method: str = 'attention_rescoring',
                 force_yes: bool = False,
                 punc_model_name: str = 'ernie_linear_p3_wudao',
                 punc_config: Optional[os.PathLike] = None,
                 punc_ckpt_path: Optional[os.PathLike] = None,
                 punc_vocab: Optional[os.PathLike] = None):
        asr_executor = ASRExecutor()
        text_executor = TextExecutor()
        paddle.set_device(paddle.get_device())
        asr_executor._init_from_path(asr_model_name, 'en', sample_rate, asr_config, decode_method, asr_ckpt_path)
        text_executor._init_from_path('punc', punc_model_name, 'en', punc_config, punc_ckpt_path, punc_vocab)
        self.asr_executor = asr_executor
        self.text_executor = text_executor
        self.asr_model_name = asr_model_name
        self.sample_rate = sample_rate
        self.asr_config = asr_config
        self.asr_ckpt_path = asr_ckpt_path
        self.decode_method = decode_method
        self.force_yes = force_yes
        self.punc_model_name = punc_model_name
        self.punc_config = punc_config
        self.punc_ckpt_path = punc_ckpt_path
        self.punc_vocab = punc_vocab
        self.blank_signal_level = -5.0  # per ms
        self.blank_signal_min_time = 300  # ms
        self.BLANK_SIGNAL_MIN_GAP = 10  # just 4 convenience
        self.BLANK_SIGNAL_PER_CHANNEL = 10.0  # based on the paddlespeech model & the sample, it means 10ms per channel

    def preprocess4asr(self, audio_fpath: os.PathLike):
        print("Altered By Iris Xu For Debug")
        print("Preprocess audio_file:", audio_fpath)
       # assert "deepspeech2online" not in self.asr_model_name and "deepspeech2offline" not in self.asr_model_name
        print("get the preprocess conf")
        preprocess_conf = self.asr_executor.config.preprocess_config
        preprocess_args = {"train": False}
        preprocessing = Transformation(preprocess_conf)
        print("read the audio file")
        audio, audio_sample_rate = soundfile.read(audio_fpath, dtype="int16", always_2d=True)

        if audio.shape[1] >= 2:
            print("merge the audio sample into mono track")
            audio = audio.mean(axis=1, dtype=numpy.int16)
        else:
            audio = audio[:, 0]

        if self.sample_rate != audio_sample_rate:
            print("change the audio sample rate")
            # pcm16 -> pcm 32
            audio = self.asr_executor._pcm16to32(audio)
            audio = librosa.resample(audio, audio_sample_rate, self.sample_rate)
            # pcm32 -> pcm 16
            audio = self.asr_executor._pcm32to16(audio)

        print(f"audio shape: {audio.shape}")
        # fbank
        audio = preprocessing(audio, **preprocess_args)
        print(f"audio shape after preprocessing: {audio.shape}")

        audio = paddle.to_tensor(audio, dtype='float32').unsqueeze(axis=0)
        print(f"audio feat shape: {audio.shape}")
        time_len = audio.shape[1]
        print("time_len", time_len)
        audio_sum = paddle.sum(audio, axis=-1).tolist()[0]
        cnt = 0
        stime_cnt = 0.0
        i_cnt = 0
        blank_signal_level_per_gap = self.blank_signal_level * self.BLANK_SIGNAL_PER_CHANNEL * self.BLANK_SIGNAL_MIN_GAP
        blank_signal_min_gap_num = self.blank_signal_min_time / self.BLANK_SIGNAL_PER_CHANNEL / self.BLANK_SIGNAL_MIN_GAP
        should_try_start_record = True
        audio_snippets = []
        for i in range(0, time_len, self.BLANK_SIGNAL_MIN_GAP):  # Here 10 is just the gap
            stime_now = i * self.BLANK_SIGNAL_PER_CHANNEL  # Here 10.0 is 10 ms
            etime_now = min(i + self.BLANK_SIGNAL_MIN_GAP, time_len) * self.BLANK_SIGNAL_PER_CHANNEL
            sum_v = sum(audio_sum[i: i + self.BLANK_SIGNAL_MIN_GAP])
            should_add = i + self.BLANK_SIGNAL_MIN_GAP >= time_len and cnt <= self.blank_signal_min_time and not should_try_start_record
            if sum_v < blank_signal_level_per_gap:
                cnt += 1
                if cnt == blank_signal_min_gap_num and not should_try_start_record:
                    should_add = True
            else:
                cnt = 0
            if should_add:
                assert stime_cnt < etime_now
                assert i_cnt < i + self.BLANK_SIGNAL_MIN_GAP
                if etime_now - stime_cnt > 2000:
                    audio_snippets.append({"start_time_ms": stime_cnt, "end_time_ms": etime_now,
                                           "audio": paddle.slice(audio, axes=[1], starts=[i_cnt],
                                                                 ends=[min(i + self.BLANK_SIGNAL_MIN_GAP, time_len)])})
                    should_try_start_record = True
            if sum_v >= blank_signal_level_per_gap and should_try_start_record:
                stime_cnt = stime_now
                i_cnt = i
                should_try_start_record = False
        if len(audio_snippets) == 0:
            print("[WARN] Cannot cut out any snippets, please adjust self.blank_signal_level")
        return audio_snippets

    @staticmethod
    def timems2srttimestr(t_in_ms):
        # 02:27:33,861
        t = int(t_in_ms)
        t_ms = t % 1000
        t /= 1000
        t_s = t % 60
        t /= 60
        t_min = t % 60
        t /= 60
        assert t <= 24
        return "%02d:%02d:%02d,%03d" % (t, t_min, t_s, t_ms)

    def process_one(self, audio_file: os.PathLike):
        audio_file = os.path.abspath(audio_file)
        # TODO: stepwise input
        self.asr_executor._check(audio_file, self.sample_rate, self.force_yes)
        audio_snippets = self.preprocess4asr(audio_file)
        for audio_snippet in audio_snippets:
            self.asr_executor._inputs["audio"] = audio_snippet["audio"]
            self.asr_executor._inputs["audio_len"] = paddle.to_tensor(audio_snippet["audio"].shape[1])
            self.asr_executor.infer(self.asr_model_name)
            text = self.asr_executor.postprocess()  # Retrieve result of asr.
            if len(text) > 0:
                self.text_executor.preprocess(text)
                self.text_executor.infer()
                punced_text = self.text_executor.postprocess()  # Retrieve result of text task.
                audio_snippet["text"] = punced_text
            else:
                audio_snippet["text"] = punced_text = ""
            print('ASR Result: \n{}'.format(text))
            print('Text Result: \n{}'.format(punced_text))
        with codecs.open(os.path.join("out", os.path.basename(audio_file) + ".srt"), "w", encoding="utf-8") as fout:
            for i, audio_snippet in enumerate(audio_snippets):
                fout.write(f"{i + 1}\n")
                fout.write("{} --> {}\n".format(self.timems2srttimestr(audio_snippet["start_time_ms"]),
                                                self.timems2srttimestr(audio_snippet["end_time_ms"])))
                fout.write(audio_snippet["text"])
                fout.write("\n\n")


if __name__ == "__main__":
    producer = MAutomaticVideo2MRTProducer()
    producer.process_one(audio_file=os.path.abspath(os.path.expanduser(args.input)))
