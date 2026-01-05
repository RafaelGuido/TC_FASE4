# 🧠 Análise de Vídeo

Este projeto é a nossa entrega referente ao tech challenge da FASE 4 da Pós-Graduação em IA para Devs pela FIAP, focado em processamento de vídeo, reconhecimento facial, análise de emoções e inferência de atividades.

## 🎯 Objetivo do Projeto

O objetivo principal foi criar uma aplicação que processasse um vídeo pré-estabelezido para este desafio (`Unlocking Facial Recognition_ Diverse Activities Analysis.mp4`), a fim de executar as seguintes tarefas:

1.  **Reconhecimento Facial (Tarefa 1):** Detectar e demarcar todos os rostos presentes nos quadros do vídeo.
2.  **Análise de Expressões Emocionais (Tarefa 2):** Classificar a emoção dominante (Ex: Feliz, Triste, Raiva, etc...) de cada rosto detectado.
3.  **Detecção de Atividades (Tarefa 3):** Detectar as atividades humanas (Ex: Trabalhando/Estudando, Comunicando, Lendo, etc...) a partir dos objetos detectados.
4.  **Geração de Resumo (Tarefa 4):** Compilar um resumo automático e quantitativo das emoções e atividades mais frequentes presentes no vídeo.

## 🚀 Tecnologias Utilizadas

| Tecnologia | Função Principal | Modelo/Abordagem |
| :--- | :--- | :--- |
| **OpenCV (`cv2`)** | Manipulação e I/O de vídeo e imagem. | BGR e Funções de Desenho |
| **FER** | Detecção de Rosto e Classificação de Emoções. | CNN Pré-treinada (Baseada em MTCNN) |
| **Ultralytics** | Detecção de Objetos (Proxy para Atividades). | YOLOv8n (YOLOv8 Nano) |
| **Numpy/Collections** | Estruturas de dados e contadores para agregação de resumo. | Contagem de Ocorrências |

## 🤖 Otimização de Desempenho e Informações Adicionais
Para evitar lentidão, a análise de emoções e atividades é realizada por amostragem:
- Análise de Emoções: A cada 5 quadros (FRAME_SKIP_EMOTION).
- Análise de Atividades: A cada 10 quadros (FRAME_SKIP_ACTIVITY).
- Velocidade de Exibição: Foi limitada a ~33 FPS para visualização legível da análise em tempo real.
- Uma janela do OpenCV será aberta após executar o arquivo ```facial_recognition.py```, exibindo a análise em tempo real (Rostos: verde, Atividades: roxo).
- A aplicação criará uma pasta chamada rostos_detectados_e_atividades com os recortes dos rostos, com o nome dos arquivos contendo o frame e a emoção detectada.
- Um Resumo da Análise detalhado (Tarefa 4) será impresso no console ao final da execução, além de salvarmos em um arquivo separado chamado analise_final.txt.

## 🧐 Decisões de Design e Justificativa Técnica

### Escolha da Biblioteca de Análise de Emoções: `FER` vs. `deepface`

Para a implementação das Tarefas 1 e 2, optamos pela biblioteca **`FER` (Face Emotion Recognition)** em vez de soluções mais abrangentes como o `deepface`.

Nossa decisão foi guiada por priorizar a **simplicidade** e a **performance** no pipeline de processamento de vídeo, mantendo o foco estrito nos requisitos do desafio.

| Critério | `FER` (Escolha) | `deepface` (Alternativa vista nas aulas) | Justificativa |
| :--- | :--- | :--- | :--- |
| **Integração em Vídeo** | Designada para retornar dados de emoção por *frame* de forma clara e direta. | Exige manipulação mais complexa da estrutura de resultados por quadro. | **Simplicidade e Velocidade de Integração** no loop do OpenCV. |
| **Foco** | Estrito em **Detecção de Rosto e Emoção**. | Abrangente, incluindo Emoção, Idade, Gênero e Raça. | A complexidade extra foi evitada para otimizar o **tempo de execução** do projeto, focando apenas no requisito de emoção. |

## ⚙️ Como Executar

### Pré-requisitos

Certifique-se de ter o Python 3.8+ instalado e os seguintes arquivos na sua pasta de projeto:
* `facial_recognition.py` (O código principal)
* `Unlocking Facial Recognition_ Diverse Activities Analysis.mp4` (O vídeo de entrada)
* `yolov8n.pt` (O arquivo de pesos do modelo YOLOv8 Nano)

### Instalação das Dependências

Instale todas as bibliotecas necessárias usando o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```