from File_Converter_Factory import FileConverterFactory
from processing_facade import PolynomialFacade
from preprocessoring_strategy import PolynomialRegressionPreprocessor
from spacy_nlp import SpacyNLP
import pandas as pd
from pathlib import Path

def main():
    # המרת קובץ ל־DataFrame
    path = Path("C:/Users/444/Downloads/Food_Delivery_Times.csv")
    converter = FileConverterFactory().get(path)
    csv_path = converter.convert_to_csv(path)
    df = pd.read_csv(csv_path)


    # בחר מודל: Polynomial Regression
    preprocessor = PolynomialRegressionPreprocessor(degree=2)
    model = PolynomialFacade(preprocessor)

    results = model.train_and_evaluate(df, target_col="Delivery_Time_min")  # שים את שם העמודה הרלוונטית
    print(results)

    model.plot()
    model.get_optimal_x()

    # הדגמה של NLP
    sentence = "Taylor Swift performed in Los Angeles on March 3rd, 2023."
    nlp = SpacyNLP()
    print(nlp.get_ents(sentence))
    print(nlp.get_persons(sentence))
    print(nlp.get_lemmas(sentence))

if __name__ == "__main__":
    main()
