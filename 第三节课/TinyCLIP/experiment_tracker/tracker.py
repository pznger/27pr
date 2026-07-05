"""
可复用的实验追踪模块，基于 SwanLab。
复用到 BLIP / LLaVA 等项目时，复制整个 experiment_tracker/ 目录即可。

用法：
    from experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker(
        project="CLIP",
        config={"lr": 1e-3, "epochs": 5},
    )
    tracker.log({"train_loss": 0.5, "epoch": 1})
    tracker.finish()

API Key 配置（三选一）：
    方式1（推荐）：run.sh 中 SWANLAB_API_KEY
    方式2：本目录 secrets.py 中写入
    方式3：环境变量 SWANLAB_API_KEY 或 swanlab login
"""
import os


def _setup_keys():
    """从 secrets.py 读 API key 并设为环境变量。"""
    try:
        from . import secrets
        if getattr(secrets, "SWANLAB_API_KEY", ""):
            os.environ.setdefault("SWANLAB_API_KEY", secrets.SWANLAB_API_KEY)
    except ImportError:
        pass


class ExperimentTracker:
    """SwanLab 实验追踪封装：init / log / finish。"""

    def __init__(self, project: str, config: dict = None, backends: list = None):
        _setup_keys()
        self._sw_run = None

        try:
            import swanlab
            self._sw_run = swanlab.init(project=project, config=config)
            print(f"[tracker] SwanLab 已初始化，project={project}")
        except Exception as e:
            print(f"[tracker] SwanLab 初始化失败: {e}，跳过")

    def log(self, metrics: dict, step: int = None):
        if self._sw_run is not None:
            try:
                import swanlab
                swanlab.log(metrics, step=step)
            except Exception:
                pass

    def finish(self):
        if self._sw_run is not None:
            try:
                import swanlab
                swanlab.finish()
            except Exception:
                pass

    @property
    def active(self) -> bool:
        return self._sw_run is not None
