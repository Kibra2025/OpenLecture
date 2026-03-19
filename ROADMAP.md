# Roadmap Tecnica OpenLecture

Data: 2026-03-19

## Obiettivo

Rendere OpenLecture piu affidabile, piu efficiente e piu semplice da evolvere senza introdurre regressioni.

Ordine di lavoro consigliato:

1. stabilizzare il progetto
2. ridurre gli sprechi di RAM e I/O
3. conservare piu struttura nel transcript
4. ottimizzare solo dopo avere misure reali

## Fase 0 - Baseline Affidabile - Fatto

Obiettivo: mettere in sicurezza il progetto prima di fare refactor piu profondi.

- [ ] Task 0.1: creare una cartella `tests/` e aggiungere test per `output_formatter.py`
  Files: `tests/test_output_formatter.py`, `openlecture/output_formatter.py`
  Done when: il formatter e coperto almeno per transcript vuoto, transcript normale e input non stringa.

- [ ] Task 0.2: aggiungere test CLI senza caricare Whisper reale
  Files: `tests/test_cli.py`, `openlecture/cli.py`
  Done when: sono coperti `--help`, path output di default, `--output` esplicito ed errore su input mancante.

- [ ] Task 0.3: aggiungere test di validazione per audio path e chunk size
  Files: `tests/test_audio_utils.py`, `openlecture/audio_utils.py`, `openlecture/transcribe.py`
  Done when: sono coperti path vuoto, file inesistente, path non-file e `chunk_length_ms <= 0`.

- [ ] Task 0.4: aggiungere CI minima su GitHub Actions
  Files: `.github/workflows/ci.yml`
  Done when: ad ogni push girano test e installazione base almeno su Windows e Linux.

- [ ] Task 0.5: allineare dipendenze e packaging
  Files: `pyproject.toml`, `requirements.txt`
  Done when: le dipendenze dichiarate non sono in conflitto e `ffmpeg-python` viene o rimosso o motivato chiaramente.

- [ ] Task 0.6: sistemare l'entrypoint CLI
  Files: `pyproject.toml`, `openlecture/cli.py`
  Done when: il comportamento pubblico della CLI e coerente con quello che si vuole supportare davvero.

Nota: questa fase ha la priorita piu alta. Senza di lei, ogni refactor successivo diventa fragile.

## Fase 1 - Hardening UX ed Errori - Fatto

Obiettivo: rendere il tool piu prevedibile e piu facile da diagnosticare.

- [ ] Task 1.1: introdurre messaggi di errore piu specifici
  Files: `openlecture/transcribe.py`, `openlecture/audio_utils.py`, `openlecture/cli.py`
  Done when: errori come file mancante, decode fallita, modello non caricabile e output non scrivibile sono distinguibili.

- [ ] Task 1.2: aggiungere una modalita `--verbose`
  Files: `openlecture/cli.py`, `openlecture/transcribe.py`
  Done when: in modalita normale l'errore e pulito, in modalita verbose si vede anche la causa originale.

- [ ] Task 1.3: validare meglio il path di output
  Files: `openlecture/cli.py`
  Done when: il tool gestisce in modo chiaro cartelle mancanti, path invalidi e permessi insufficienti.

- [ ] Task 1.4: uniformare l'output console
  Files: `openlecture/cli.py`, `openlecture/transcribe.py`
  Done when: si usa uno stile unico per progressi, warning ed errori, senza mix casuale di `print()` e CLI output.

## Fase 2 - Efficienza a Basso Rischio - Fatto

Obiettivo: ridurre RAM, I/O su disco e overhead inutile senza cambiare troppo l'architettura.

- [ ] Task 2.1: evitare il chunking per file brevi
  Files: `openlecture/transcribe.py`
  Done when: i file piccoli possono essere trascritti direttamente senza passare dal pipeline di chunk export.

- [ ] Task 2.2: processare un chunk alla volta
  Files: `openlecture/audio_utils.py`, `openlecture/transcribe.py`
  Done when: il programma non crea piu la lista completa di chunk esportati prima di iniziare a trascrivere.

- [ ] Task 2.3: cancellare ogni chunk subito dopo l'uso
  Files: `openlecture/transcribe.py`
  Done when: il picco di spazio temporaneo su disco e limitato al chunk corrente o quasi.

- [ ] Task 2.4: aggiungere benchmark minimi
  Files: `scripts/benchmark.py` oppure `tests/perf/`, README o doc dedicata
  Done when: esiste almeno un modo ripetibile per misurare durata totale, throughput e overhead dei chunk.

Nota: questa e la fase con il miglior ritorno pratico sulle performance.

## Fase 3 - Transcript Strutturato - fatto

Obiettivo: smettere di perdere informazione utile troppo presto.

- [ ] Task 3.1: introdurre una struttura dati per i segmenti
  Files: `openlecture/transcribe.py` o nuovo modulo `openlecture/models.py`
  Done when: esiste almeno un tipo esplicito con `start`, `end`, `text`.

- [ ] Task 3.2: far ritornare segmenti da `_transcribe_file()`
  Files: `openlecture/transcribe.py`
  Done when: la trascrizione non viene appiattita subito in una sola stringa.

- [ ] Task 3.3: far ritornare un risultato piu ricco da `transcribe_audio()`
  Files: `openlecture/transcribe.py`, `openlecture/cli.py`
  Done when: il livello alto puo scegliere come renderizzare il transcript senza perdere metadata.

- [ ] Task 3.4: aggiornare il formatter Markdown
  Files: `openlecture/output_formatter.py`
  Done when: il Markdown viene generato partendo da segmenti strutturati, non da una stringa piatta.

- [ ] Task 3.5: aggiungere almeno un output strutturato
  Files: `openlecture/output_formatter.py`, `openlecture/cli.py`
  Done when: oltre al Markdown esiste almeno un formato come JSON.

Nota: questa fase aumenta sia affidabilita logica sia spazio di evoluzione futura.

## Fase 4 - Controlli Utente e Qualita Trascrizione

Obiettivo: dare controllo ai tradeoff velocita/qualita e migliorare i casi reali.

- [ ] Task 4.1: esporre `--model`
  Files: `openlecture/cli.py`, `openlecture/transcribe.py`
  Done when: il modello non e piu hardcoded a `medium`.

- [ ] Task 4.2: esporre `--beam-size`, `--device`, `--compute-type`
  Files: `openlecture/cli.py`, `openlecture/transcribe.py`
  Done when: l'utente puo adattare il tool a CPU, GPU e obiettivi di velocita.

- [ ] Task 4.3: esporre `--language`
  Files: `openlecture/cli.py`, `openlecture/transcribe.py`
  Done when: l'utente puo forzare la lingua quando l'autodetect e inaffidabile.

- [ ] Task 4.4: sperimentare overlap o split piu intelligenti
  Files: `openlecture/audio_utils.py`, `openlecture/transcribe.py`
  Done when: i bordi tra chunk sono meno soggetti a tagliare parole o frasi.

## Fase 5 - Progress Bar Robusta

Obiettivo: mantenere una buona UX senza legarsi troppo a dettagli interni fragili.

- [ ] Task 5.1: aggiungere test per la progress bar custom
  Files: `tests/test_progress.py`, `openlecture/transcribe.py`
  Done when: il comportamento base non rompe facilmente dopo refactor interni.

- [ ] Task 5.2: introdurre un fallback semplice
  Files: `openlecture/transcribe.py`
  Done when: se il monkey-patching di `faster_whisper.transcribe.tqdm` non e disponibile, il tool continua comunque a funzionare.

- [ ] Task 5.3: decidere se tenere o semplificare la progress bar avanzata
  Files: `openlecture/transcribe.py`
  Done when: il team ha scelto consapevolmente tra UX ricca e manutenzione piu semplice.

## Cose Da Non Fare Subito

- non partire dalla parallelizzazione dei chunk
- non aggiungere una GUI prima di avere una CLI stabile
- non aggiungere nuove feature di output prima di avere test di base
- non complicare troppo il formatter finche il transcript viene ancora appiattito presto

## Roadmap Breve Consigliata

Se l'obiettivo e fare progressi rapidi senza perdersi:

1. completare tutta la Fase 0
2. completare almeno Task 1.1, 1.2 e 1.3
3. completare Task 2.1, 2.2 e 2.3
4. solo dopo iniziare la Fase 3

## Definizione Di Successo

La roadmap sta funzionando se, alla fine delle prime tre fasi:

- il progetto ha test automatici
- la CLI e coerente e prevedibile
- gli errori sono piu chiari
- il consumo di disco e memoria per file lunghi si riduce
- la pipeline non perde piu subito tutta la struttura del transcript
