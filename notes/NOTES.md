## ML Basics — Dataset & Split

### X vs y
- X: matrice delle feature (righe per feature)
- y: target
- Shape tipiche: X = (n_samples, n_features), y = (n_samples,)

### Perché stampo shape
- per eseguire sanity check

### Train/Test split
- Scopo: Dividere il dataset e allenarlo su una parte per testare l'efficacia del modello sulla seconda
- Regola: il test set è “esame” -> non usarlo per decisioni di training

### random_state
- costante randomizzatrice per dividere sempre allo stesso modo le righe di un dataframe tra training e test 

### max_iter
- costante passata per definire il numero massimo di iterazioni che il modello dispone per convergere sulla soluzione. Più è alto più do tempo al modello di avvicinarsi ad una soluzione ottimale. Se lo do basso rischio che si fermi prima di averla trovata

### Pitfall (unire test al training)
- Test contamination ≠ overfitting automatico, ma invalida la valutazione. Non è sempre overfitting. Non sempre l'esisto avrà accuracy del 100%. Ma test conteminato

### Valutazioni su Seed e test_size per il calcolo della varianza
- Ho valutato la stabilità della baseline ripetendo l’esperimento su più seed per diversi test_size, riportando mean/std/min/max dell’accuracy.
- All’aumentare di test_size (training più piccolo) l’accuracy media cala e la varianza aumenta: il risultato dipende di più dallo split (“fortuna”).
- stratify=y migliora sia performance media sia stabilità (std/min migliori), perché mantiene proporzioni simili delle classi tra train e test.