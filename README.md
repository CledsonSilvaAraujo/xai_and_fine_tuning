# Chest X-Ray — ResNet-50, Grad-CAM++ e XAI-Guided Fine-Tuning

Projeto da disciplina de visão computacional: classificação binária **NORMAL vs PNEUMONIA** no dataset [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), transfer learning em **ResNet-50**, explicabilidade com **Grad-CAM++** e refinamento guiado por atenção (**XAI-guided fine-tuning**).

---

## Passo a passo — como rodar o projeto

Siga a ordem abaixo para reproduzir o pipeline completo, do ambiente até a apresentação final.

### Passo 1 — Preparar o ambiente

Crie e ative um ambiente virtual Python na raiz do projeto, instale as dependências e verifique se tudo está correto:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
python scripts/verify_environment.py
```

**O que você obtém:** confirmação de que PyTorch, torchvision e as demais bibliotecas estão prontas para treinar e gerar mapas Grad-CAM++.

---

### Passo 2 — Obter e validar os dados

O código procura o dataset em `data/chest_xray/` ou `src/data/chest_xray/`, com as subpastas `train/{NORMAL,PNEUMONIA}` e `test/{NORMAL,PNEUMONIA}`. Se ainda não tiver os dados, use `scripts/download_chest_xray_kaggle.py` (requer credenciais do Kaggle configuradas).

Em seguida, valide o pipeline de leitura e pré-processamento:

```bash
python scripts/smoke_data_pipeline.py
```

**O que você obtém:** certeza de que o DataLoader carrega as imagens (224×224, normalização ImageNet, augmentation no treino) antes de iniciar o treinamento.

---

### Passo 3 — Treinar o baseline (head only)

A primeira estratégia treina apenas a camada totalmente conectada (`fc`), mantendo o backbone ImageNet congelado. É o baseline rápido e clássico de transfer learning.

```bash
python scripts/train_baseline.py --strategy head_only --epochs 5
```

Por padrão usa learning rate `1e-3`, `WeightedRandomSampler` para balancear classes, `CrossEntropyLoss` com pesos inversamente proporcionais à frequência, seed 42 e salva o **melhor checkpoint pela AUC de validação** (não necessariamente a última época).

**O que você obtém:**
- `checkpoints/resnet50_head_only.pt`
- `results/metrics_head_only.json`

---

### Passo 4 — Treinar o full fine-tune

A segunda estratégia descongela `layer3`, `layer4` e `fc`, adaptando camadas mais profundas ao domínio das radiografias. O restante do backbone permanece congelado.

```bash
python scripts/train_baseline.py --strategy full_finetune --epochs 10
```

O learning rate padrão é `1e-4` (backbone com LR 10× menor). Recomenda-se **10 épocas** — são ~22 M parâmetros treináveis e a convergência é mais lenta que no head only.

**O que você obtém:**
- `checkpoints/resnet50_full_finetune.pt`
- `results/metrics_full_finetune.json`

Caminhos customizados (opcional):

```bash
python scripts/train_baseline.py --strategy head_only \
  --checkpoint checkpoints/meu_head.pt \
  --metrics-out results/meu_metricas.json
```

---

### Passo 5 — Grad-CAM++ em lote e análise de erros (full finetune)

Com o checkpoint do full finetune, gere mapas Grad-CAM++ para todo o conjunto de teste e categorize os erros (FP/FN) com o `overlap_score` — fração da ativação **fora** da máscara pulmonar proxy (crop central de 60%).

```bash
python scripts/gradcam_error_analysis.py \
  --checkpoint checkpoints/resnet50_full_finetune.pt \
  --out-dir results/gradcam \
  --error-json results/error_analysis.json
```

Para teste rápido, adicione `--max-images 50`.

**O que você obtém:**
- Overlays em `results/gradcam/{TP,TN,FP,FN}/`
- `results/error_analysis.json` com categorias `focused` (<0,2), `partial` (0,2–0,5) e `distracted` (>0,5)

Comandos opcionais para comparar dois modelos na mesma imagem:

```bash
python scripts/compare_gradcam.py \
  --checkpoint-a checkpoints/resnet50_head_only.pt \
  --checkpoint-b checkpoints/resnet50_full_finetune.pt \
  --image caminho/para/imagem.jpeg \
  --combined

python scripts/gradcam_preview.py \
  --checkpoint checkpoints/resnet50_full_finetune.pt \
  --image caminho/para/imagem.jpeg \
  --out results/gradcam_um_modelo.png
```

---

### Passo 6 — XAI-Guided Fine-Tuning V1

Partindo do full finetune, o script calcula Grad-CAM++ em imagens de **treino**, define pesos por amostra (`w = 1 + α × overlap_score`, com α=2,0) e faz fine-tuning guiado por atenção.

```bash
python scripts/xai_guided_finetune.py \
  --checkpoint checkpoints/resnet50_full_finetune.pt \
  --epochs 3 --lr 1e-5 --alpha 2.0 \
  --checkpoint-out checkpoints/resnet50_xai_guided.pt \
  --metrics-out results/metrics_xai_guided.json
```

No WSL2, o script usa `num_workers=0` automaticamente para evitar erro de shared memory no DataLoader.

**O que você obtém:**
- `checkpoints/resnet50_xai_guided.pt`
- `results/metrics_xai_guided.json`
- `results/xai_cam_weights.json`

---

### Passo 7 — Grad-CAM++ no modelo XAI-Guided V1

Repita a análise de erros apontando para o checkpoint XAI-guided, para comparar como a atenção evoluiu.

```bash
python scripts/gradcam_error_analysis.py \
  --checkpoint checkpoints/resnet50_xai_guided.pt \
  --out-dir results/gradcam_xai \
  --error-json results/error_analysis_xai.json
```

**O que você obtém:** overlays e `results/error_analysis_xai.json`.

---

### Passo 8 — XAI-Guided Fine-Tuning V2 (segunda iteração)

Partindo do V1, com learning rate ainda menor (`5e-6`), tenta-se reduzir os falsos negativos remanescentes.

```bash
python scripts/xai_guided_finetune.py \
  --checkpoint checkpoints/resnet50_xai_guided.pt \
  --epochs 3 --lr 5e-6 --alpha 2.0 \
  --checkpoint-out checkpoints/resnet50_xai_guided_v2.pt \
  --metrics-out results/metrics_xai_guided_v2.json \
  --cam-weights-json results/xai_cam_weights_v2.json
```

**O que você obtém:**
- `checkpoints/resnet50_xai_guided_v2.pt`
- `results/metrics_xai_guided_v2.json`

Opcional — Grad-CAM++ no V2:

```bash
python scripts/gradcam_error_analysis.py \
  --checkpoint checkpoints/resnet50_xai_guided_v2.pt \
  --out-dir results/gradcam_xai_v2 \
  --error-json results/error_analysis_xai_v2.json
```

---

### Passo 9 — Figuras comparativas (4 modelos)

Com os JSONs de métricas e análise de erro gerados, produza as figuras para artigo e apresentação:

```bash
python scripts/compare_attention.py --max-grid 6
```

**O que você obtém em `figures/`:**
- `comparison_gradcam_3models.pdf`
- `overlap_score_distribution.pdf`
- `metrics_4models.pdf`
- `metrics_evolution.pdf`

Figuras individuais do full finetune (matriz de confusão, ROC, barras de erro):

```bash
python scripts/generate_figures.py \
  --metrics results/metrics_full_finetune.json \
  --error-json results/error_analysis.json \
  --out-dir figures
```

---

### Passo 10 — Gerar as apresentações

**Apresentação intermediária (entrega 2):**

```bash
python scripts/create_presentation.py --out apresentacao.pptx
```

**Apresentação final (entrega 3 — conclusão dos 4 modelos):**

```bash
python scripts/create_presentation_final.py --out apresentacao_final.pptx
```

O roteiro de fala slide a slide está em `roteiro_apresentacao.md`.

---

### Passo 11 — Compilar o artigo (ABNT)

O artigo está em `artigo_abnt.tex` (classe abntex2), com referências em `referencias.bib`:

```bash
pdflatex artigo_abnt.tex
bibtex artigo_abnt
pdflatex artigo_abnt.tex
pdflatex artigo_abnt.tex
```

Alternativa: use o [Overleaf](https://www.overleaf.com) colando o `.tex` e o `.bib`. As figuras em `figures/` são incluídas via `\includegraphics`.

---

## Resumo da ordem de execução

1. Ambiente → 2. Dados → 3. Head only → 4. Full finetune → 5. Grad-CAM++ (full) → 6. XAI V1 → 7. Grad-CAM++ (V1) → 8. XAI V2 → 9. Figuras → 10. Apresentações → 11. Artigo

---

## Resultados de referência (conjunto de teste, 624 imagens)

No conjunto de teste, o modelo **head only** (`metrics_head_only.json`) obteve **87,18%** de acerto e **12,82%** de erro, com AUC-ROC de **0,9300** e F1 de **0,8942** (FP=28, FN=52).

O **full fine-tune** (`metrics_full_finetune.json`) alcançou **86,54%** de acerto e **13,46%** de erro, com AUC-ROC de **0,9693** e F1 de **0,9019** (FP=80, FN=4) — maior recall, porém muitos falsos positivos.

O **XAI-Guided V1** (`metrics_xai_guided.json`) atingiu **91,03%** de acerto, AUC **0,9663**, F1 **0,9300** (FP=38, FN=18) — melhor equilíbrio geral, com redução de 52,5% nos FP em relação ao full finetune.

O **XAI-Guided V2** (`metrics_xai_guided_v2.json`) obteve **90,06%** de acerto, F1 **0,9248** (FP=53, FN=9) — mais sensível (menos FN), ao custo de mais FP que o V1.

---

## O que vai para os arquivos de métricas

Cada execução de treino grava um JSON em `results/` com:

- **`accuracy_percent`**: porcentagem de predições corretas (acerto global).
- **`error_percent`**: porcentagem de predições erradas (100 − acerto).
- **`auc_roc`**, **`f1`**, **`precision`**, **`recall`**
- **`confusion_matrix`**: formato scikit-learn com classes `[0, 1]` → **`[[TN, FP], [FN, TP]]`** (linhas = verdade: 0=NORMAL, 1=PNEUMONIA).

Há blocos **`validation`** (15% do treino) e **`test`** (pasta `test/` oficial).

---

## Estratégias de treino

Existem duas estratégias base, escolhidas pelo parâmetro `--strategy`:

Na estratégia **head_only**, o backbone pré-treinado no ImageNet permanece congelado e apenas a camada `fc` é ajustada (~4.098 parâmetros). É rápida e serve como baseline. Checkpoint: `checkpoints/resnet50_head_only.pt`. Métricas: `results/metrics_head_only.json`.

Na estratégia **full_finetune**, são descongeladas `layer3`, `layer4` e `fc` (~22 M parâmetros), enquanto Conv1, Layer1 e Layer2 permanecem congelados. Checkpoint: `checkpoints/resnet50_full_finetune.pt`. Métricas: `results/metrics_full_finetune.json`.

As estratégias **XAI-Guided V1 e V2** partem dos checkpoints anteriores e aplicam pesos por amostra derivados do Grad-CAM++ — não usam `--strategy`, e sim `scripts/xai_guided_finetune.py`.

---

## Estrutura do projeto

```
checkpoints/              # pesos (.pt) — ignorados pelo git (grandes)
results/
  metrics_*.json          # métricas finais (versionados)
  error_analysis*.json    # análise de erros Grad-CAM++ (versionados)
  gradcam/                # overlays — ignorados pelo git
  gradcam_xai/            # overlays XAI V1 — ignorados pelo git
  gradcam_xai_v2/         # overlays XAI V2 — ignorados pelo git
figures/                  # PDFs gerados — ignorados pelo git
scripts/                  # treino, Grad-CAM++, XAI, apresentações
src/
  data/chest_xray.py      # DataModule e transforms
  evaluation.py           # acurácia, matriz de confusão, AUC
artigo_abnt.tex           # artigo ABNT (fonte)
referencias.bib           # bibliografia
roteiro_apresentacao.md   # roteiro de fala
```

---

## Arquivos ignorados pelo git

O `.gitignore` exclui artefatos gerados automaticamente: apresentações `.pptx`, PDF compilado do LaTeX, pasta `figures/`, overlays Grad-CAM++, logs e pesos CAM intermediários. Os JSONs de métricas e análise de erro permanecem versionados por serem o registro numérico do experimento.
