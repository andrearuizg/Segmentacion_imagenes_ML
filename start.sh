echo "GENERAR DATASET"
python3 gen_dataset.py
echo "ENTRENAMIENTO SVM"
python3 train_svm.py > data_svm.txt
echo "PREDICCIÓN SVM"
python3 verification_svm.py
