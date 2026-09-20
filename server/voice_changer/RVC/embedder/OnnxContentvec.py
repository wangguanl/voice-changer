import numpy as np
import torch
from torch import device

from voice_changer.RVC.embedder.Embedder import Embedder


class OnnxContentvec(Embedder):
    """local-custom implementation of the ONNX contentvec embedder.

    pretrain/content_vec_500.onnx exposes:
      input  audio [1, T] float32 (16kHz waveform)
      output units9 [1, T', 256]   (layer9 + final_proj, v1 models)
      output unit12 [1, T', 768]   (layer12 raw, v2 models)
      output unit12s [1, T', 768]  (layer12 variant)
    The original stub raised "Not implemented", which silently forced the
    fairseq fallback (unavailable on Windows/torch2.0 venvs).
    """

    def loadModel(self, file: str, dev: device) -> Embedder:
        super().setProps("hubert_base", file, dev, False)
        import onnxruntime

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if dev.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.model = onnxruntime.InferenceSession(file, providers=providers)
        return self

    def extractFeatures(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        wav = feats.detach().cpu().numpy().astype(np.float32)
        if wav.ndim == 1:
            wav = wav[None, :]
        if embOutputLayer == 9:
            out_name = "units9"
        elif useFinalProj:
            out_name = "unit12s"
        else:
            out_name = "unit12"
        outs = self.model.run([out_name], {"audio": wav})
        return torch.tensor(np.asarray(outs[0])).to(self.dev)
