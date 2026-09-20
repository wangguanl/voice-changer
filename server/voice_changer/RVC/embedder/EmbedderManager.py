from torch import device

from const import EmbedderType
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.embedder.OnnxContentvec import OnnxContentvec
from voice_changer.RVC.embedder.Whisper import Whisper
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


class EmbedderManager:
    currentEmbedder: Embedder | None = None
    params: VoiceChangerParams

    @classmethod
    def initialize(cls, params: VoiceChangerParams):
        cls.params = params

    @classmethod
    def _fairseq(cls, which: str):
        # local-custom: lazy imports — fairseq is optional (ONNX contentvec
        # path works without it); an eager top-level import broke slots
        import importlib

        mod = importlib.import_module(f"voice_changer.RVC.embedder.{which}")
        return getattr(mod, which)

    @classmethod
    def getEmbedder(cls, embederType: EmbedderType, isHalf: bool, dev: device) -> Embedder:
        if cls.currentEmbedder is None:
            print("[Voice Changer] generate new embedder. (no embedder)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)
        elif cls.currentEmbedder.matchCondition(embederType) is False:
            print("[Voice Changer] generate new embedder. (not match)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)
        else:
            print("[Voice Changer] generate new embedder. (anyway)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)
        return cls.currentEmbedder

    @classmethod
    def loadEmbedder(cls, embederType: EmbedderType, isHalf: bool, dev: device) -> Embedder:
        if embederType == "hubert_base":
            try:
                if cls.params.content_vec_500_onnx_on is False:
                    raise Exception("[Voice Changer][Embedder] onnx is off")
                file = cls.params.content_vec_500_onnx
                return OnnxContentvec().loadModel(file, dev)
            except Exception as e:  # noqa
                print("[Voice Changer] use torch contentvec", e)
                file = cls.params.hubert_base
                return cls._fairseq("FairseqHubert")().loadModel(file, dev, isHalf)
        elif embederType == "hubert-base-japanese":
            file = cls.params.hubert_base_jp
            return cls._fairseq("FairseqHubertJp")().loadModel(file, dev, isHalf)
        elif embederType == "contentvec":
            try:
                if cls.params.content_vec_500_onnx_on is False:
                    raise Exception("[Voice Changer][Embedder] onnx is off")
                file = cls.params.content_vec_500_onnx
                return OnnxContentvec().loadModel(file, dev)
            except Exception as e:
                print(e)
                file = cls.params.hubert_base
                return cls._fairseq("FairseqContentvec")().loadModel(file, dev, isHalf)
        elif embederType == "whisper":
            file = cls.params.whisper_tiny
            return Whisper().loadModel(file, dev, isHalf)
        else:
            file = cls.params.hubert_base
            return cls._fairseq("FairseqHubert")().loadModel(file, dev, isHalf)
