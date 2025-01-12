
import rl_zoo3
import rl_zoo3.train
from rl_zoo3.enjoy import enjoy
from sbx import SAC, TQC

from Models.EnTRPO.EnTRPO import EnTRPO, sample_entrpo_params
from Models.EnTRPOR.EnTRPOR import EnTRPOR, sample_entrpor_params
from Models.TRPOQ.TRPOQ import TRPOQ, sample_trpoq_params
from Models.TRPOQ.TRPOQ2 import TRPOQ2
from Models.TRPOR.TRPOR import TRPOR, sample_trpor_params

models = {
    "entrpo": {"model": EnTRPO, "sample": sample_entrpo_params},
    "trpoq": {"model": TRPOQ, "sample": sample_trpoq_params},
    "trpoq2": {"model": TRPOQ2, "sample": sample_trpoq_params},
    "trpor": {"model": TRPOR, "sample": sample_trpor_params},
    "entrpor": {"model": EnTRPOR, "sample": sample_entrpor_params},
    "sac": {"model": SAC},
    "tqc": {"model": TQC},
}

for model_name, value in models.items():
  model_class = value["model"]
  sample = value["sample"] if "sample" in value else None

  rl_zoo3.ALGOS[model_name] = model_class
  rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER[model_name] = sample


rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

if __name__ == "__main__":

  enjoy()
