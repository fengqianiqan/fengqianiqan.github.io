---
title: MiniMax H3 × ComfyUI：H200 上的音视频生成与推理加速
tags:
  - MiniMax H3
  - ComfyUI
  - 推理加速
categories:
  - 工程实践
abbrlink: e851fb7f
date: 2026-08-13 18:00:00
---
# MiniMax H3 × ComfyUI：H200 上的音视频生成与推理加速

这是一篇关于 MiniMax H3 的 ComfyUI 推理展示。实验使用 NVIDIA H200 单卡生成 768×1344、362 帧、24 FPS 的音视频内容，并比较不同推理加速组合的实际效果。

> **结果摘要：** 非 Sage 路径中，TE-Speed + EasyCache 将端到端时间从 19 分 43 秒缩短到 7 分 44 秒，达到 2.55× 加速；SageAttention 可以单独或搭配 EasyCache 完成推理，但当前与 TE-Speed 组合时稳定性不足。

本次工作围绕以下开源项目展开：

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)：推理工作流框架；
- [MiniMax H3](https://www.modelscope.cn/models/MiniMax/MiniMax-H3/)：音视频联合生成模型；
- [TE-Speed-MiniMaxH3-OSS](https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS)：MiniMax H3 推理加速方案；
- [SageAttention](https://github.com/thu-ml/SageAttention)：Attention 算子加速方案；
- [EasyCache](https://github.com/H-EmbodVis/EasyCache)：扩散采样缓存加速方案。

## 环境与推理配置

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA H200，单卡 |
| PyTorch | 2.9.1+cu128 |
| CUDA Runtime | 12.8 |
| ComfyUI | 0.30.1 |
| Triton | 3.5.1 |
| SageAttention | 2.2.0 |
| comfy-aimdo | 0.4.11 |
| comfy-kitchen | 0.2.26 |
| 分辨率 | 768×1344 |
| 视频长度 | 362 帧，24 FPS |
| 采样步数 | 20 steps |
| 随机种子 | 42 |
| Sampler / Scheduler | `res_multistep` / `simple` |

### 关于 CUDA 优化路径

本次推理仍然运行在 H200 GPU 上，Sage 配置也确认启用了 SageAttention 的 CUDA kernel；但启动日志提示，当前 PyTorch 为 `cu128`，而 comfy-kitchen 的优化 CUDA/Triton 算子要求 `cu130+`。因此，comfy-kitchen 的 CUDA 和 Triton 后端在本次测试中处于禁用状态，相关操作回退到 eager 实现。

这不会使本次结果失效，但意味着表中的耗时代表“当前实际环境”的性能，而不是所有优化 CUDA 算子完整启用后的理论最佳性能。后续升级到版本匹配的 PyTorch/CUDA 环境后，需要重新测试才能进行公平比较。

## 使用的模型

| 用途 | 模型文件 |
| --- | --- |
| T2VA 主模型 | `minimax_h3_fl2va_pruned_bf16.safetensors` |
| 文本编码器 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| 视频 VAE | `minimax_h3_video_vae_fp16.safetensors` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` |

主模型以 BF16 运行，文本编码器采用 NVFP4/AWQ 量化，视频 VAE 为 FP16，音频 VAE 为 FP32。

## Prompt

原始分镜描述经过 MiniMax H3 的 Context-IR 前处理，整理为同时覆盖画面、动作、镜头和声音的实际推理 prompt。内容是一位老年图书修复师在安静修复室中修补古籍的 15 秒纪实镜头。

<details>
<summary><strong>展开查看实际推理 Prompt</strong></summary>

```text
integrated_multimodal_description: [Shot 1] Cinematic, close-up shot with a slight handheld camera shake and a shallow depth of field, focusing on a wooden table in a quiet restoration room. Soft, natural afternoon window light spills from the left. An on-screen elderly male book restorer is mending a yellowed ancient book. His visible hands have thick finger joints, with tiny pieces of paper and dried paste stains embedded in the nail beds. He wears a gray cotton and linen work uniform with dark blue sleeves. His right hand uses metal tweezers to pick up a delicate, thin piece of repair paper featuring manually torn frayed edges and clearly visible fibers. He gently lays the paper over a damaged area of the ancient page, while the fingertips of his left hand press down on the patch with a steady, gentle motion. The camera smoothly pedestals up and tilts up, shifting focus from his hands to his face. The man has short gray hair with the temples shaved very short, and his face is marked by deep creases and age spots. His gaze is lowered, carefully observing the fit of the patch through a circular magnifying glass held near his eye. The natural light reveals intricate facial textures: real skin sagging, fine stubble, faint capillaries, and asymmetric eyebrow bones. He maintains a calm expression, blinking extremely slowly. He then straightens his upper body, takes the magnifying glass away from his face, and gently rubs the bridge of his nose with his right thumb and index finger, his shoulders dropping slightly in a subtle reaction of fatigue. Lowering his head again to inspect the mended page, his mouth twitches slightly at the corner, revealing a barely perceptible look of quiet relief.

overall_soundscape: A hushed room tone establishes the quiet environment, punctuated by the audible, crisp rustle of thin paper fibers and the faint metallic click of tweezers. As the left hand presses the patch, a soft, dry rubbing sound is clearly heard against the ancient page. The continuous subtle rustle of heavy cotton and linen fabric accompanies his posture adjustments as he straightens up. A distinct skin-on-skin friction sound occurs as he rubs the bridge of his nose, followed by a soft, pronounced exhalation of breath.

non_diegetic_music: N/A
```

</details>

## 加速配置与测试方法

| 组件 | 参数 |
| --- | --- |
| TE-Speed | `processing_control_value=0.12`、窗口 `10%–90%`、`mcs=2`、`device=auto`、`cache_depth=0.75` |
| EasyCache | `reuse_threshold=0.10`、窗口 `15%–95%` |
| SageAttention | ComfyUI 服务使用 `--use-sage-attention` |
| 非 Sage Attention | ComfyUI 服务使用 `--use-pytorch-cross-attention` |

所有配置使用相同的 prompt、seed、分辨率、帧数、采样器和步数。Sage 与非 Sage 使用独立服务，ComfyUI 以 `--cache-none` 启动，避免相同输入命中节点结果缓存。表中记录单次正式运行的采样时间和端到端时间，因此结果适合展示当前环境下的实际趋势，不应视为多次重复统计后的通用 benchmark。

## 推理结果

### 非 Sage 路径

| 配置 | 采样时间 | 端到端时间 | 相对 Baseline | 结果 |
| --- | ---: | ---: | ---: | --- |
| Baseline | 19 分 07 秒 | 19 分 43 秒 | 1.00× | 完成 |
| TE-Speed | 10 分 23 秒 | 10 分 58 秒 | 1.80× | 完成 |
| EasyCache | 13 分 22 秒 | 13 分 58 秒 | 1.41× | 完成 |
| TE-Speed + EasyCache | 7 分 08 秒 | 7 分 44 秒 | 2.55× | 完成 |

在这次测试中，TE-Speed + EasyCache 获得了最快的完整推理结果，端到端耗时相比 Baseline 减少约 60.8%。

### SageAttention 路径

| 配置 | 采样时间 | 端到端时间 | 结果 |
| --- | ---: | ---: | --- |
| SageAttention | 9 分 53 秒 | 10 分 30 秒 | 完成 |
| EasyCache + SageAttention | 7 分 25 秒 | 8 分 02 秒 | 完成 |
| TE-Speed + SageAttention | — | — | 未完成 |
| TE-Speed + EasyCache + SageAttention | — | — | 未完成 |

SageAttention 单独使用，以及与 EasyCache 组合使用时可以完成推理；加入 TE-Speed 后，当前组合仍存在稳定性问题。因此，现阶段更适合使用非 Sage 的 TE-Speed + EasyCache 作为可运行配置，Sage 相关组合仍处于实验阶段。

> **结果边界：** 这组测试聚焦速度和运行完成情况，没有进行 SSIM、PSNR 或音频响度评估。因此，“最快”只表示耗时最低，不等同于生成质量最佳。

## 视频展示

下面展示本次测试中全部 6 个成功生成的视频。每个视频均可通过自身控制条单独播放，以便清楚对比画面与音频效果。

<div class="h3-video-wall">
<div class="h3-video-grid">
<article class="h3-video-card"><div class="h3-video-card-title"><span>Baseline</span><small>19 分 43 秒</small></div><video controls preload="metadata" playsinline data-h3-video><source src="/video/non-sage/t2va/base_00001_.mp4" type="video/mp4"></video></article>
<article class="h3-video-card"><div class="h3-video-card-title"><span>TE-Speed</span><small>10 分 58 秒 · 1.80×</small></div><video controls preload="metadata" playsinline data-h3-video><source src="/video/non-sage/t2va/te_00001_.mp4" type="video/mp4"></video></article>
<article class="h3-video-card"><div class="h3-video-card-title"><span>EasyCache</span><small>13 分 58 秒 · 1.41×</small></div><video controls preload="metadata" playsinline data-h3-video><source src="/video/non-sage/t2va/easycache_00001_.mp4" type="video/mp4"></video></article>
<article class="h3-video-card"><div class="h3-video-card-title"><span>TE-Speed + EasyCache</span><small>7 分 44 秒 · 2.55×</small></div><video controls preload="metadata" playsinline data-h3-video><source src="/video/non-sage/t2va/te_easycache_00001_.mp4" type="video/mp4"></video></article>
<article class="h3-video-card"><div class="h3-video-card-title"><span>SageAttention</span><small>10 分 30 秒</small></div><video controls preload="metadata" playsinline data-h3-video><source src="/video/sage/t2va/sageattention_00001_.mp4" type="video/mp4"></video></article>
<article class="h3-video-card"><div class="h3-video-card-title"><span>EasyCache + SageAttention</span><small>8 分 02 秒</small></div><video controls preload="metadata" playsinline data-h3-video><source src="/video/sage/t2va/easycache_sageattention_00001_.mp4" type="video/mp4"></video></article>
</div>
</div>

<style>
  .h3-video-wall { margin: 1.5rem 0; }
  .h3-video-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
  .h3-video-card { overflow: hidden; border: 1px solid rgba(58, 74, 92, .13); border-radius: .8rem; background: rgba(255, 255, 255, .72); box-shadow: 0 .25rem .9rem rgba(30, 43, 61, .07); }
  .h3-video-card-title { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; padding: .7rem .8rem; font-weight: 650; font-size: .95rem; }
  .h3-video-card-title small { flex: none; color: #7a8491; font-size: .76rem; font-weight: 500; white-space: nowrap; }
  .h3-video-card video { display: block; width: 100%; aspect-ratio: 9 / 16; background: #111; object-fit: contain; }
  @media (max-width: 900px) { .h3-video-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 560px) { .h3-video-grid { grid-template-columns: 1fr; } .h3-video-card-title { font-size: 1rem; } }
</style>

TE-Speed + SageAttention 以及 TE-Speed + EasyCache + SageAttention 未生成可展示的视频。

## 结语

这次测试表明，在当前 H200 环境中，ComfyUI 可以完成 MiniMax H3 的高分辨率音视频生成；TE-Speed 和 EasyCache 组合能够显著减少推理时间。SageAttention 单独也可以运行，但与 TE-Speed 叠加后仍需要进一步解决稳定性问题。

速度之外，最终配置还需要同时满足生成质量、运行稳定性和可复现性。后续将继续完善视频展示，并对 SageAttention 与 TE-Speed 的组合进行单变量排查。
