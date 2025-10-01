```markdown
# Solução de Problemas

## ❌ trtexec não encontrado
```bash
export PATH=/usr/src/tensorrt/bin:$PATH
```

## ❌ Out of Memory durante build
- Aumente o swap: `sudo swapon /swapfile`
- Reduza o workspace: `--workspace=1024`

## ❌ Baixo FPS
- Verifique modo de performance: `sudo nvpmodel -m 0`
- Confirme clocks: `sudo jetson_clocks`
- Verifique temperaturas: `tegrastats`

## ❌ Câmera não detectada
```bash
ls /dev/video*
sudo usermod -a -G video $USER
```