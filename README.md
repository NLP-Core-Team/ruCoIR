# ruCoIR Датасет

ruCoir датасет переведен с использованием phi4 на русский. Передены некоторые задания: apps, codefeedback-st, stackoverflow-qa, cosqa, codesearchnet.

## Запуск замеров

Для запуска замеров выполните следующую команду:
```bash
python sentence_transformermers_run_eval.py /\
  --model_name ai-forever/FRIDA \
  --tasks apps codefeedback-st stackoverflow-qa cosqa codesearchnet \
  --batch_size 128 \
  --hf_token hf_...
```
Результаты сохраняются в папке `results`.

Для замера по Api Voyager. URL захардкожен.

```bash
python API_retrival_run_eval.py \
  --model_name voyage-code-3\
  --tasks apps codefeedback-st stackoverflow-qa cosqa codesearchnet \
  --batch_size 128 \
  --hf_token hf_...
```
## Чтение результатов

Чтобы прочитать результаты, выполните одну из следующих команд:

- Для чтения результатов для конкретной модели:
  python read_scores.py ./results/model_name

- Для чтения всех замеров по всем моделям:
  python read_scores_all.py ./results. Результат будет сохранен в файл `results.csv`.

## Результаты замеров открытых моделей
| Model                                   | Number of Params (B) | Mean | apps.  | codefeedback-st. | stackoverflow-qa. | cosqa.  | CodeSearchNet-go. | CodeSearchNet-java. | CodeSearchNet-javascript. | CodeSearchNet-ruby. | CodeSearchNet-python. | CodeSearchNet-php. |
|-----------------------------------------|----------------------|------|--------|-------------------|-------------------|---------|--------------------|----------------------|-----------------------------|----------------------|------------------------|---------------------|
| intfloat/multilingual-e5-small          | 0.1                  | 0.52 | 0.12   | 0.71              | 0.82              | 0.26    | 0.54               | 0.48                 | 0.44                        | 0.54                 | 0.69                   | 0.62                |
| Alibaba-NLP/gte-modernbert-base         | 0.1                  | 0.63 | 0.36   | 0.74              | 0.88              | 0.21    | 0.75               | 0.68                 | 0.59                        | 0.68                 | 0.59                   | 0.81                |
| fyaronskiy/code_retriever_ru_en         | 0.1                  | 0.66 | 0.14   | 0.81              | 0.81              | 0.25    | 0.76               | 0.74                 | 0.57                        | 0.80                 | **0.91**               | 0.83                |
| Alibaba-NLP/gte-multilingual-base       | 0.3                  | 0.71 | 0.14   | 0.84              | 0.86              | 0.26    | 0.85               | 0.81                 | 0.75                        | 0.79                 | 0.89                   | 0.89                |
| Salesforce/SFR-Embedding-Code-400M_R    | 0.4                  | 0.42 | 0.16   | 0.44              | 0.67              | 0.09    | 0.67               | 0.38                 | 0.38                        | 0.45                 | 0.47                   | 0.53                |
| intfloat/multilingual-e5-large          | 0.6                  | 0.67 | 0.22   | 0.85              | 0.91              | 0.36    | 0.75               | 0.70                 | 0.57                        | 0.76                 | 0.79                   | 0.76                |
| intfloat/multilingual-e5-large-instruct | 0.6                  | 0.65 | 0.25   | 0.85              | 0.93              | 0.36    | 0.71               | 0.66                 | 0.48                        | 0.70                 | 0.71                   | 0.80                |
| codesage/codesage-large-v2              | 0.6                  | 0.67 | 0.45   | 0.68              | 0.80              | 0.18    | 0.82               | 0.77                 | 0.73                        | 0.85                 | 0.79                   |-                    |
| ai-forever/FRIDA                        | 0.8                  | 0.60 | 0.04   | 0.69              | 0.71              | 0.19    | 0.79               | 0.70                 | 0.57                        | 0.71                 | 0.79                   | 0.78                |
| Alibaba-NLP/gte-Qwen2-1.5B-instruct     | 1.5                  | 0.69 | 0.28   | 0.91              | 0.91              | 0.35    | 0.65               | 0.79                 | 0.64                        | 0.76                 | 0.78                   | 0.87                |
| Salesforce/SFR-Embedding-Code-2B_R      | 2                    | 0.80 | 0.85   | 0.92              | 0.93              | 0.38    | 0.85               | 0.76                 | 0.70                        | **0.89**             | 0.87                   | 0.82                |
| BAAI/bge-code-v1                        | 2                    | 0.83 | **0.96**| **0.96**          | 0.94              | 0.30    | **0.87**           | **0.88**             | **0.78**                    | **0.90**             | 0.83                   | **0.91**            |
| ai-sage/Giga-Embeddings-instruct        | 3                    | 0.58 | 0.20   | 0.87              | 0.91              | 0.32    | 0.73               | 0.61                 | 0.40                        | 0.61                 | 0.62                   | 0.56                |
| intfloat/e5-mistral-7b-instruct         | 7                    | 0.62 | 0.23   | 0.87              | **0.95**          | 0.32    | 0.74               | 0.63                 | 0.49                        | 0.67                 | 0.66                   | 0.59                |
| voyage-code-3                           | -                    | 0.69 | 0.43   | 0.92              | 0.92              | **0.47**| —                  | —                    | —                           | —                    | —                      | —                   |

