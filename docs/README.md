# GitHub Pages — `liaa-allissonchervenski.github.io`

Esta pasta (`docs/`) contém a estrutura do site estático para apresentação do projeto no GitHub Pages.

---

## Estrutura de Arquivos

```
docs/
├── index.html        # Página principal com todas as seções estruturais
├── css/
│   └── style.css     # Estilos CSS estruturais com variáveis CSS para fácil customização
├── assets/           # Imagens, diagramas e logotipos (para inclusão posterior)
└── README.md         # Documentação de publicação e execução
```

---

## Seções Contempladas no HTML

1. **Header & Hero:** Identificação do projeto, LIAA, autor e metadados.
2. **1. Contexto do Projeto:** Fundamentos de sEMG, próteses mioelétricas e restrições de TinyML no ESP32-S3.
3. **2. O Problema Abordado:** Gargalo dos 70% de acurácia, crosstalk muscular e inviabilidade de CNNs pesadas em microcontroladores.
4. **3. Proposta de Solução:** Pipeline de DSP causal (DF2T), extração de características Hudgins TD e Arquitetura Hierárquica Coarse-to-Fine (M11) com gatilho de confiança (&tau; = 0.80).
5. **4. Evolução no Semestre:** Linha do tempo das 5 fases de desenvolvimento (Governança SDD/TDD &rarr; DSP &rarr; Baselines &rarr; Quantização/Canais &rarr; Modelos Híbridos & Hardware QEMU).
6. **5. Resultados Consolidados:** Tabela com métricas completas de acurácia, fast path, latência e energia por janela de inferência.

---

## Como Publicar no GitHub Pages

### Opção 1: Diretamente a partir deste repositório (`liaa-allissonchervenski`)
1. No repositório GitHub, acesse **Settings** &rarr; **Pages**.
2. Em **Build and deployment** &rarr; **Source**, selecione **Deploy from a branch**.
3. Escolha a branch `gh-pages` (pasta `/(root)`) — ou a branch de trabalho e a pasta **/docs**.
4. Clique em **Save**. O site será publicado em `https://AllissonChervenski.github.io/liaa-allissonchervenski/`.

### Opção 2: No repositório de organização / usuário `liaa-allissonchervenski.github.io`
1. Crie o repositório `liaa-allissonchervenski.github.io` no GitHub.
2. Copie o conteúdo de `docs/*` (`index.html`, `css/`, `assets/`) para a raiz desse repositório.
3. O site será servido diretamente em `https://liaa-allissonchervenski.github.io/`.

---

## Como Visualizar Localmente

Você pode abrir o arquivo `docs/index.html` diretamente em qualquer navegador, ou iniciar um servidor local simples via Python:

```bash
python3 -m http.server --directory docs 8000
```
Em seguida, acesse no navegador: `http://localhost:8000`.
