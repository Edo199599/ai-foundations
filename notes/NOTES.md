## ML Basics — Dataset & Split

### X vs y
- X: matrice delle feature (righe per feature)
- y: target
- Shape tipiche: X = (n_samples, n_features), y = (n_samples,)

### Perché stampo shape
- per eseguire sanity check

### Train/Test split
- Scopo:
- Regola: il test set è “esame” -> non usarlo per decisioni di training

### random_state
- costante randomizzatrice per dividere sempre allo stesso modo le righe di un dataframe tra training e test 

### Pitfall (unire test al training)
- Test contamination ≠ overfitting automatico, ma invalida la valutazione. Non è sempre overfitting. Non sempre l'esisto avrà accuracy del 100%. Ma test conteminato
