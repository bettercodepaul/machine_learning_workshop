import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

def assert_approx(actual, expected, tol=0.001):
    assert abs(actual - expected) < abs(tol*expected)

def train_fit_predict_score(model, df):
    X = df.drop(columns=["Preis", "Nachbarschaft"])
    y = df.get_column("Preis")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = round(metrics.r2_score(y_test, y_pred), 12)
    return (model, score)

class HintSolution:

    def __init__(self, question, check, hint, solution):
        self.tries = 1
        self._question = question
        self._hint = hint
        self._check = check
        self._solution = solution

    def question(self):
        print(self._question)

    def hint(self):
        print(self._hint)

    def solution(self):
        print(self._solution)

    def check(self, *args):
        if type(args[0]) == type(Ellipsis):
            print("❓ Moment, die drei Punkte musst du mit deiner Lösung ersetzen!")
            return
        try:
            self._check(*args)
            check_result = True
        except:
            check_result = False
        
        if check_result:
            if self.tries == 1:
                #print("✅ Wow, first try and you nailed it! You're a natural problem-solver! 🎉👏")
                print("✅ Wow, erster Versuch - du hast es voll drauf! Du bist ein geborener Problemlöser! 🎉👏")
            elif self.tries == 2:
                #print("✅ Right on target! You hit the bullseye on your second shot! 🎯👏")
                print("✅ Voll ins Schwarze getroffen! Schon beim zweiten Schuss mitten auf die Zwölf! 🎯👏")
            elif self.tries == 3:
                #print("✅ Persistence pays off! Third try's a charm, and you did it! 🌟👏")
                print("✅ Durchhalten zahlt sich aus! Mit dem dritten Versuch hast du es geschafft! 🌟👏")
            else:
                #print("✅ It might have taken a few tries, but you're unstoppable! 😅👊")
                print("✅ Es hat vielleicht ein paar Versuche gebraucht, aber du bist nicht aufzuhalten! 😅👊")
                
        else:
            if self.tries == 1:
                #print("🤔 Give it another shot! You're just getting started. 🔍")
                print("🤔 Probier es nochmal! Du hast ja gerade erst angefangen. 🔍")
            elif self.tries == 2:
                #print("🤔 Two tries down, but the solution is within reach. Keep going! 🧐")
                print("🤔 Zwei Versuche hinter dir, aber die Lösung ist in Reichweite. Nur Mut! 🧐")
            elif self.tries == 3:
                #print("🤔 Almost there, just one more push! You can do it! 😬")
                print("🤔 Du bist fast am Ziel, nur noch eine letzte Anstrengung! Du schaffst das! 😬")
            else:
                #print("🤔 It's tough, but don't lose hope! Maybe consider using the hint() method now? 😓")
                print("🤔 Es ist schwierig, aber verlier nicht die Hoffnung! Hast du schon die hint() Methode verwendet? 😓")
            self.tries = self.tries + 1


def q0_check(x):
    assert x == "BettercallPaul"

q0 = HintSolution(
    'Bei welcher Firma arbeiten die meisten von uns?',
    q0_check,
    'Es ist nicht BettercallSaul.',
    'solution = "BettercallPaul"'
)

def q1_check(x):
    assert len(x.filter(pl.col("Wohnflaeche").ge(4000))) == 0
    assert len(x) == 2925

q1 = HintSolution(
    'Es gibt ein paar Häuser mit sehr hoher Wohnfläche. Entferne alle Häuser mit mehr als 4000 Quadratfuß Wohnfläche.',
    q1_check,
    'Benutze die filter Methode und die lt (less than) Methode auf der Spalte Wohnflaeche.',
    'q1_df = df.filter(pl.col("Wohnflaeche").lt(4000))'
)

def q2_check(X_train_Zustand_median, y_train_median, X_test_Zustand_median, y_test_median):
    assert_approx(X_train_Zustand_median, 5.0)
    assert_approx(y_train_median, 160000.0)
    assert_approx(X_test_Zustand_median, 5.0)
    assert_approx(y_test_median, 165000.0)

q2 = HintSolution(
    'Ermittel den Median-Wert jeweils auf den Trainings- und Testdaten für den Zustand und den Verkaufspreis.',
    q2_check,
    'Benutze die Methode describe(), schreibe die passenden Werte ("50%") ab und weise sie den Variablen zu.',
    'X_train_Zustand_median = 5.0\ny_train_median = 160000.0\nX_test_Zustand_median = 5.0\ny_test_median = 165000.0'
)

def q3_check(y_test, y_pred):
    r2 = metrics.r2_score(y_test, y_pred)
    assert r2 > 0.764
    assert r2 < 0.99 # nicht schummeln mit y_pred=y_test!

q3 = HintSolution(
    'Erstelle ein Modell, das den Verkaufspreis besser vorhersagt, in dem du mehr Features als nur "Nachbarschaft" weglässt.',
    q3_check,
    'Prüfe die Features "Zimmer" und "Verkaufsjahr" genauer.',
    'X = df.drop(columns=["Preis", "Nachbarschaft", "Zimmer", "Verkaufsjahr"])\ny = df.get_column("Preis")\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nmetrics.r2_score(y_test, y_pred)'
)

def q4_check(df, r2_before, r2_after):
    assert "Verkaufsalter" in df.columns
    assert "Baujahr" in df.columns
    assert "Verkaufsjahr" in df.columns
    assert len(df.filter(pl.col("Verkaufsalter").ne(pl.col("Verkaufsjahr")-pl.col("Baujahr"))))==0
    assert_approx(r2_before, r2_after, tol=0.000000001)

q4 = HintSolution(
    'Erstelle jetzt ein neues Feature mit dem Namen "Verkaufsalter", das das Alter des Hauses beim Verkauf bestimmt.\nWie ändert sich dadurch die Vorhersage? Wird sie besser oder schlechter?\nWarum?',
    q4_check,
    'Das Verkaufsalter lässt sich bestimmen mit pl.col("Verkaufsjahr")-pl.col("Baujahr")',
    'df_after = pl.read_csv("AmesHousing.csv")\ndf_after = df_after.with_columns(\n    (pl.col("Verkaufsjahr")-pl.col("Baujahr")).alias("Verkaufsalter")\n)'
)

def q5_check(model):
    assert type(model).__name__=="RandomForestRegressor" or type(model).__name__=="GradientBoostingRegressor"

q5 = HintSolution(
    'Das lineare Modell ist sehr einfach. Mit anderen Verfahren kannst Du bessere Ergebnisse erzielen.\nProbiere auch `sklearn.ensemble.RandomForestRegressor`, `sklearn.ensemble.GradientBoostingRegressor` und `sklearn.neighbors.KNeighborsRegressor` aus.\nWomit erreichst Du die besten Vorhersagen?',
    q5_check,
    'Du musst die Modelle importieren, z.B. mit `from sklearn.neighbors import KNeighborsRegressor` und die Variable setzen `q5_model = KNeighborsRegressor()\n. Dann das Modell mit der Hilfsmethode `train_fit_predict_score` trainieren und testen.',
    '# Random Forest ist auf diesen Daten in 75% der Fälle besser als Gradient Boosting\nfrom sklearn.ensemble import RandomForestRegressor\ndf = pl.read_csv("AmesHousing.csv")\nq5_model = RandomForestRegressor()\n_, q5_r2 = train_fit_predict_score(q5_model, df)\nprint(q5_r2)'
)

def q6_check(best_encoding_lr, best_encoding_rf):
    assert best_encoding_lr == "Dummy Encoding"
    assert best_encoding_rf == "Target Encoding"

q6 = HintSolution(
    'Probiere die unterschiedlichen Encodings mit dem Random Forest und dem linearen Modell aus.',
    q6_check,
    'Für Integer Encoding und Dummy Encoding kannst Du direkt die Methode train_fit_predict_score benutzen.\nFür Target Encoding musst du das Target Encoding an X_train und X_test joinen, das ursprüngliche Feature Nachbarschaft dann entfernen und Training und Vorhersage selber implementieren.',
    'from sklearn.ensemble import RandomForestRegressor\nfor model in [LinearRegression(), RandomForestRegressor()]:\n    model_name = type(model).__name__\n    _, r2_int = train_fit_predict_score(model, df_int_encoding)\n    _, r2_dummy = train_fit_predict_score(model, df_dummy_encoding)\n    X_train_target_encoded = X_train.join(target_encoding, on="Nachbarschaft").drop(columns="Nachbarschaft")\n    model.fit(X_train_target_encoded, y_train)\n    y_pred = model.predict(X_test.join(target_encoding, on="Nachbarschaft").drop(columns="Nachbarschaft"))\n    r2_target = metrics.r2_score(y_test, y_pred)\n    print(f"{model_name} mit Integer Encoding: {r2_int:.3f}")\n    print(f"{model_name} mit Dummy Encoding: {r2_dummy:.3f}")\n    print(f"{model_name} mit Target Encoding: {r2_target:.3f}")\nq6_best_encoding_linear_regression = "Dummy Encoding"\nq6_best_encoding_random_forest = "Target Encoding"'
)

def q7_check(problem):
    assert problem == "Qualitaet*Zustand"

q7 = HintSolution(
    'Probiere verschiedene Klassifikationen aus jeweils mit der Zielspalte "Nachbarschaft", "Zimmer" und "Qualitaet" aus. Warum ist die Accuracy für "Qualitaet" so hoch?',
    q7_check,
    'Ein bestimmtes Feature sorgt dafür, dass es sehr einfach für das Modell ist, die Qualitaet vorherzusagen. Gebe den Namen dieses Features an.',
    'problem="Qualitaet*Zustand"' 
)

def q8_check(width, height):
    assert width == 240
    assert height == 240

q8 = HintSolution(
    'Prüfe, welche Breit und Höhe die Bilder im Batch haben, damit du das MLP richtig konfigurieren kannst.',
    q8_check,
    'Hole dir einen Batch mit `X, y = dls.one_batch() und prüfe die Dimensionen mit `X.shape`.',
    'X, y = dls.one_batch()\nq8_width = X.shape[2] # 240\nq8_height = X.shape[3] # 240'
)

def q9_check(output_size):
    assert output_size == 15

q9 = HintSolution(
    'Prüfe, wie viele Output-Neuronen das MLP haben muss.',
    q9_check,
    'Zähle die Anzahl der Klassen im Ordner mercedes-12k/train oder nutze das Vokabular des Dataloaders.',
    'q9_output_size = len(dls.vocab)'
)

def q10_check(model):
    assert model.width == 240
    assert model.height == 240
    assert model.output_size == 15

q10 = HintSolution(
    'Erstelle ein Modell mit den richtigen Dimensionen für die WhichCar-Bilder. Nutze für die Hidden Layers jeweils 16 Neuronen.',
    q10_check,
    'Erstelle eine Instanz der Klasse SimpleMlp mit den passenden Größen (siehe auch q8 und q9).',
    'q10_model = SimpleMlp(240, 240, 16, 16, 15)'
)

def q11_check(learn):
    assert learn.n_epoch == 1
    assert learn.metrics[0].value > 0.0
    assert learn.metrics[0].value < 0.2

q11 = HintSolution(
    'Trainiere das Modell jetzt für eine Epoche mit der initiale Lernrate 0.01',
    q11_check,
    'Nutze die Methode `fit_one_cycle` und trainiere wirklich nur für eine Epoche.',
    'q11_learn = Learner(\n    dls,\n    model=SimpleMlp(240, 240, 16, 16, 15),\n    loss_func=nn.CrossEntropyLoss(),\n    metrics=accuracy\n)\nq11_learn.fit_one_cycle(1, 0.01)'
)