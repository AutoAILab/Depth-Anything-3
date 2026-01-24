---
description: Installation and setup instructions for Depth Anything 3
---

Follow these steps to set up the environment:

1. **Basic Installation**
   ```bash
   pip install xformers torch>=2 torchvision
   pip install -e .
   ```

2. **Gradio App Support (Optional)**
   ```bash
   pip install -e ".[app]"
   ```

3. **Full Installation**
   ```bash
   pip install -e ".[all]"
   ```

4. **Gaussian Head Support**
   ```bash
   pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
   ```

---

**Note**: Python 3.10+ is recommended for the Gradio app.
