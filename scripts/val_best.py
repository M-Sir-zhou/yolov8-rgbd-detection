import sys

import torch

p = r"D:\ProjectCode\PyCharm\ultralytics-main\runs\detect\train\weights\best_4ch.pt"  # 替换为你的实际路径


def safe_torch_load(path):
    # 先尝试常规加载（weights_only=True 的新默认）
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        print(
            "普通加载失败，尝试使用 weights_only=False 并允许 ultralytics DetectionModel（仅在信任文件时使用）",
            file=sys.stderr,
        )
        # 尝试把 DetectionModel 加到 safe globals（若可用）
        try:
            import ultralytics.nn.tasks as tasks

            Det = getattr(tasks, "DetectionModel", None)
            if Det is not None:
                torch.serialization.add_safe_globals([Det])
                print("已将 ultralytics.nn.tasks.DetectionModel 加入 safe globals", file=sys.stderr)
        except Exception as ie:
            print("无法导入 ultralytics.nn.tasks 或注册 safe globals:", ie, file=sys.stderr)
        # 强制以 weights_only=False 加载（可能执行自定义代码，仅在信任文件时使用）
        return torch.load(path, map_location="cpu", weights_only=False)


ck = safe_torch_load(p)

# ck 可能是 dict、OrderedDict，或包含 'model' 键的 dict
if isinstance(ck, dict) and "model" in ck:
    model_obj = ck["model"]
else:
    model_obj = ck

# 如果 model_obj 看起来是 state_dict（dict of tensors）
state_dict = None
if isinstance(model_obj, dict):
    state_dict = model_obj
elif hasattr(model_obj, "state_dict"):
    try:
        state_dict = model_obj.state_dict()
    except Exception:
        # 尝试 attribute 转换
        state_dict = {k: v for k, v in vars(model_obj).items() if isinstance(v, torch.Tensor)}
else:
    # 最后兜底
    try:
        state_dict = dict(model_obj)
    except Exception:
        state_dict = {}

# 查找第一个 conv 权重（4维 tensor）
found = False
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor) and v.dim() == 4:
        print("first conv weight:", k, tuple(v.shape))
        found = True
        break

if not found:
    print("未在 checkpoint 中找到形状为 4 的卷积权重，checkpoint keys:", list(state_dict.keys())[:20])
