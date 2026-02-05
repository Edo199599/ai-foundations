## Week 1 — Dataset & Split

**Q1: In supervised learning, cosa rappresentano X e y?**  
A: X è la matrice delle feature (input) con shape tipica (n_samples, n_features). y è il target/label con shape tipica (n_samples). Ogni riga di X deve corrispondere alla label nello stesso indice in y.

**Q2: Perché facciamo train/test split?**  
A: Per stimare la capacità di generalizzare su dati mai visti. Se valuti sul train, la metrica è ottimistica e non rappresenta le performance reali.

**Q3: Cosa succede se usi il test set durante lo sviluppo?**  
A: Contamini il test: non è più indipendente. Anche se l’accuracy non diventa 100%, la valutazione è invalidata perché hai “guardato” l’esame.

**Q4: A cosa serve random_state in train_test_split?**  
A: A rendere lo split riproducibile: stesso random_state -> stessa suddivisione train/test, utile per debugging e confronto tra esperimenti.

**Q5: Perché X e y vanno splittati insieme?**  
A: Per mantenere l’allineamento tra input e label. Se li splitti separatamente rompi la corrispondenza e training/eval diventano privi di significato.
