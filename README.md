# Réplica: Ait Hsine y Arabo (2026)

Réplica de la arquitectura y la evaluación de:

> Ait Hsine, A., & Arabo, A. (2026). A privacy-preserving middleware architecture for detecting prompt injection and sensitive data exposure in large-language-model interactions. *Electronics, 15*(16), 3554. https://doi.org/10.3390/electronics15163554

Página oficial: https://www.mdpi.com/2079-9292/15/16/3554 (acceso abierto, licencia CC BY). Modelo base del kit Rival: https://github.com/sarthakrastogi/rival. Los autores mencionan una implementación en TypeScript, pero no publican su código.

Trabajo para el curso Propuesta de Investigación 2026-2 (Universidad de Lima), como base metodológica del proyecto M.P.P.F.I.IA.

## Qué hace

Un middleware intercepta cada prompt antes de que llegue al LLM y decide, **con señales locales**, si puede salir del equipo (Fig. 2, p. 10):

```
prompt → reglas (regex, Luhn, entropía) → heurística (H) → SLM local (3 clases)
       → ¿sensible o ataque?  sí → solo local
                              no → LLM externo (L), opcional
       → veredicto: C = 0.40·H + 0.60·L → benign | concerning | security_attack | critical_harm
       → acción:    Allow | Warn | Strong warn | Block
```

## Correspondencia con el artículo

| Archivo | Qué implementa | Dónde está en el artículo | Grado de fidelidad |
|---|---|---|---|
| `src/config.py` | IDs de recursos, plantilla del SLM, pesos, umbral, resultados de referencia | 4.4–4.10, Tablas 3, 5 y 8 (pp. 8–16) | Exacto |
| `src/rules.py` | Regex de PII y credenciales, Luhn, entropía, firmas de jailbreak | 4.4 (p. 8), 6.3 (p. 18) | **Reconstruido**: el artículo no publica las expresiones |
| `src/heuristics.py` | Indicadores de manipulación y puntaje H | 4.4 (p. 8) | **Reconstruido**: no publica los pesos |
| `src/slm.py` | Narasimha-0.6B + adaptador LoRA, clasificación generativa en CPU; línea base DeBERTa-v3 | 4.7 (pp. 9–11), 4.10 (p. 13), Tabla 5 | Exacto (modelos publicados); ID de DeBERTa asumido |
| `src/external_llm.py` | Gemini 3 Flash vía OpenRouter, temp. 0.3, 200 tokens, JSON | 4.4 (p. 8), Apéndice A (p. 22) | System prompt **parcial**: pegar el texto exacto de la Fig. A1 |
| `src/pipeline.py` | Controlador, enrutador y cascada de niveles | 4.4, 4.8, 5.4 (pp. 8–16) | Filtro exacto (5.4); cascada **reconstruida** |
| `src/metrics.py` | Accuracy con IC bootstrap, F1, FPR, FNR, recall por clase, hard negatives, Wilson | 5.1–5.4 (pp. 13–16) | Exacto |
| `src/data.py` | Carga de `v1adam/Comparison_Dataset`, con respaldo simulado | 4.9 y Data Availability (pp. 12, 21) | Exacto si carga desde Hugging Face |
| `run_experiment.py` | Ejecuta todo sobre todos los prompts y aplica el filtro después | 5.4 y 6.6 (pp. 15, 19) | Mismo procedimiento que los autores |
| `demo.py` | Middleware en vivo, prompt por prompt | Fig. 2 (p. 10) | — |

## Cómo ejecutarlo

### Opción recomendada: Google Colab (CPU)

1. Sube este repositorio a GitHub.
2. Abre `notebooks/Replica_AitHsine_Arabo_2026.ipynb` en Colab.
3. Cambia el entorno a **CPU** (el artículo mide la latencia sin GPU).
4. Ejecuta las celdas en orden.

### En tu PC

```bash
pip install -r requirements.txt
python run_experiment.py --data auto --deberta     # réplica completa en CPU
python demo.py                                     # demostración del middleware
python run_experiment.py --data sim --no-slm       # prueba rápida sin descargar modelos
```

Opciones de `run_experiment.py`:

- `--data hf|sim|auto`: dataset de Hugging Face, simulado o automático.
- `--deberta`: incluye la línea base DeBERTa-v3.
- `--llm`: activa el LLM externo; requiere `OPENROUTER_API_KEY` y tiene costo.
- `--rules-profile paper|full`: `paper` excluye los formatos que el motor de los autores no detectó (p. 18).
- `--limit N`: N prompts por clase.

### Resultados

En `results/`:

- `metricas.csv`: cada configuración junto a los valores del artículo (`paper_*`).
- `predicciones.csv`: decisión de cada componente para cada prompt.
- `enrutamiento.json`: prompts retenidos o enviados, sensibles que salieron y latencia por camino (equivalente a las Tablas 8 y 9).

## Resultados del artículo para comparar

| Configuración | Exactitud | Recall | FPR | Latencia |
|---|---|---|---|---|
| Reglas (local) | 41.67 % | 13.75 % | 2.5 % | 0.1 ms |
| SLM Narasimha-0.6B (local) | 84.17 % | 81.25 % | 10.0 % | 3 193.7 ms |
| DeBERTa-v3 PI-v2 (local) | 65.00 % | 53.75 % | 12.5 % | 146.7 ms |
| Gemini 3 Flash (remoto) | 99.17 % | 98.75 % | 0 % | 1 150.9 ms |
| **Híbrido con enrutamiento** | **95.83 %** | **100 %** | **12.5 %** | — |

Además: 70 de 120 prompts retenidos en local y 38 de 40 sensibles retenidos (Tabla 8, p. 16).

## Limitaciones de esta réplica

1. **Reglas y heurísticas reconstruidas.** Los autores describen los tipos de patrones, pero no los publican. Los resultados de las reglas pueden diferir de la Tabla 5.
2. **Sin LLM externo, el híbrido corre en "modo local".** A los prompts enviados fuera se les aplica la heurística (H ≥ 0.5) en lugar de Gemini, y así se indica en `enrutamiento.json`. Para una réplica completa del híbrido hace falta `--llm`.
3. **El dataset simulado es solo de respaldo.** Lo escribimos nosotros con la misma estructura del artículo. Como las reglas y los datos se redactaron juntos, el desempeño de las reglas sobre él es optimista. Los resultados válidos para comparar son los que usan `v1adam/Comparison_Dataset`.
4. **Nombres de columna del dataset.** El artículo no los documenta. `src/data.py` los detecta; si falla, ajústalos ahí.
5. **System prompt del LLM externo.** Copia el texto exacto de la Fig. A1 (p. 22) en `src/external_llm.py`.
6. **Línea base DeBERTa-v3.** El artículo no da el ID exacto del modelo. Se asume `protectai/deberta-v3-base-prompt-injection-v2`.
7. **Llama Guard 4 no se replica.** Es un servicio remoto de 12 mil millones de parámetros.
8. **Una sola medición de latencia**, igual que en el artículo (p. 20). Depende del CPU de Colab o de tu PC.
