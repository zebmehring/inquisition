from csv import reader
from ujson import load

with open('./squad/data/dev_eval.json') as f:
    gold_dict = load(f)

goldens = {}
for d in gold_dict.values():
    goldens[d['uuid']] = set(d['answers'])

with open('./squad/data/dev-submission.csv') as f:
    predictions = {}
    for row in f:
        uuid, answer = row.split(',', maxsplit=1)
        try:
            int(uuid, base=16)
        except:
            continue
        answer = answer.strip()
        predictions[uuid] = answer

total_incorrect = 0
empty_golden_nonempty_prediction = 0
empty_prediction_nonempty_golden = 0
bad_guess = 0
subspans = 0
for uuid, prediction in predictions.items():
    if not prediction and not goldens[uuid]:
        continue
    elif prediction in goldens[uuid]:
        continue
    total_incorrect += 1
    if not prediction or not goldens[uuid]:
        empty_golden_nonempty_prediction += 1 if prediction and not goldens[uuid] else 0
        empty_prediction_nonempty_golden += 1 if goldens[uuid] and not prediction else 0
    else:
        bad_guess += 1
        error = 'bad guess'
        if any(prediction in golden or golden in prediction for golden in goldens[uuid]):
            subspans += 1
        else:
            print(f'{error} prediction \'{prediction}\', '
                  f'expected one of {goldens[uuid]}')

print('')
print(f'total incorrect: {total_incorrect}')
print(f'empty golden nonempty prediction: {empty_golden_nonempty_prediction}')
print(f'empty prediction nonempty golden: {empty_prediction_nonempty_golden}')
print(f'bad guesses: {bad_guess}')
print(f'subspans: {subspans}')
