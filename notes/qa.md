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

**Q5: A cosa serve stratify nello split?**  
A: Permette di manterenre la stessa suddivisione tra train e test nel momendo in cui ci sono più classi. Così che non ci sia una sproporzione di alcune da falsare il risultato poi finale del test.

**Q6: Cos'è la LogisticRegression e a cosa serve max_iter?**  
A: La logisticregression è un modello di classificazione che assegna al set di caratteristiche in analisi dei pesi e ne calcola poi una possibilità di classe, itera un numero di volte fino a max_iter aggiustando quei pesi di volta in volta fino ad ottenere un modello il più tendente possibile alla soluzione ottimale minimizzando la log-loss (penalizzi molto soluzioni sicure rivelatesi sbagliate, valorizzi se giuste). Con basse max_iter rischio che il modello smetta di iterare quando non ha ancora trovato una soluzione accettabile di bilanciamento.


**Q7: Cos’è una confusion matrix?**
A: Matrice nxn dve n è il numero di classi. Ogni riga rappresenta la classe vera y_true. Le colonne rappresentano le predizioni invece fatte y_pred, quindi per ogni classe quante volte è stata predetta al posto di una differente. La diagonale rappresenta le predizioni corrette.

**Q8: “support” nel classification report cos’è?**
A: il numero di esempi veri per quella classe nel test (somma della riga nella confusion matrix).

**Q9: differenza precision vs recall in una frase?**
A: 
- precision = qualità delle predizioni di quella classe, quante volte quindi predire quella classe si è rivelato corretto; 
- recall = quanti veri di quella classe riesci a catturare, quindi quante volte del totale indovinabile è stato indovinato.

**Q10: Perché un singolo split non basta per valutare un modello?**
A: Perché la metrica può variare a seconda dello split (split “fortunato/sfortunato”). Serve stimare media e varianza su più split/seed.

**Q11: Cosa garantisce random_state?**
A: Riproducibilità dello split (e quindi dell’esperimento), non migliori performance.

**Q12: Perché usare stratify=y?**
A: Mantiene proporzioni simili delle classi tra train/test, riducendo bias e varianza soprattutto con dataset piccoli o sbilanciati.

**Q13: Se abbasso la threshold, quale tra precision e recall tende a salire per la classe 1, e perché?** 
se abbasso la soglia, il modello si sbilancia di più nell'assegnare dei positivi. Quindi aumento la recall perché perdo meno falsi negativi ma diminuisco la precisione perché aumento i falsi positivi. Quindi è più bravo a non farsi scappare record di quella classe ma rischia di prendere dentro anche elementi non di quella classe. 

**Q14: In binario, la Logistic Regression “può non assegnare nessuna classe”?**
In binario, la Logistic Regression “può non assegnare nessuna classe”? No. Mentre esamina la p della classe 1 se non supera al soglia la assegna alla classe 0 

**Q15: Che differenza c’è tra threshold in binario e scelta della classe in multi-classe (Iris)?**
Nel caso di più di 2 classi non c'è una soglia ma si tende ad assegnare alla classe con probabilità maggiore.

**Q16: Come leggo velocemente una confusion matrix in binario?**
Righe = true, colonne = pred. Diagonale = corretti. Con label [0,1] → [[TN, FP],[FN, TP]].