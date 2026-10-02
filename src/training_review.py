"""Summarise existing training histories and pilot fits without training any model.

Run: python src/training_review.py
Requires the saved protocol 2 outputs and the project's Matplotlib dependency.
"""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'logistic': 'Logistic regression', 'mlp': 'MLP', 'cnn': 'CNN'}
COLORS = ['#245B78', '#B56826']

def read(path):
    return json.loads(path.read_text())

def main():
    cfg = read(ROOT / 'config/protocol.json')
    assert cfg['protocol_version'] == 2
    selection = read(ROOT / 'outputs/selection.json')
    assert selection['protocol_sha256'] == hashlib.sha256((ROOT / 'config/protocol.json').read_bytes()).hexdigest()
    reports = ROOT / 'reports'
    figures = reports / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    pilot, final, sources = [], [], {}
    def remember(path):
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in NAMES:
        for lr in cfg['learning_rates'][name]:
            folder = ROOT / 'outputs/pilot' / name / str(lr)
            r = read(folder / 'validation.json')
            history = read(folder / 'history.json')
            for file in ['validation.json', 'history.json']:
                remember(folder / file)
            assert r['source_sha256'] == selection['source_sha256']
            assert r['protocol_sha256'] == selection['protocol_sha256']
            best, last = history[r['best_epoch'] - 1], history[-1]
            assert abs(best['validation_loss'] - r['validation_loss']) < 1e-12
            pilot.append({'model': name, 'learning_rate': lr, 'seed': r['seed'],
                          'selected': lr == selection['selected_learning_rates'][name],
                          'selected_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'],
                          'validation_accuracy': r['validation_accuracy'],
                          'validation_loss': r['validation_loss'],
                          'first_validation_accuracy': history[0]['validation_accuracy'],
                          'last_validation_accuracy': last['validation_accuracy'],
                          'last_validation_loss': last['validation_loss'],
                          'fit_seconds': r['fit_seconds_including_early_stopping_validation']})
        for condition in cfg['conditions']:
            for seed in cfg['seeds']:
                folder = ROOT / 'outputs/runs' / f'{name}_{condition}_{seed}'
                r = read(folder / 'validation.json')
                history = read(folder / 'history.json')
                for file in ['validation.json', 'history.json']:
                    remember(folder / file)
                assert r['source_sha256'] == selection['source_sha256']
                assert r['protocol_sha256'] == selection['protocol_sha256']
                best, last = history[r['best_epoch'] - 1], history[-1]
                final.append({'model': name, 'condition': condition, 'seed': seed,
                              'selected_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'],
                              'selected_train_loss': best['train_loss'], 'last_train_loss': last['train_loss'],
                              'selected_validation_loss': best['validation_loss'], 'last_validation_loss': last['validation_loss'],
                              'selected_validation_accuracy': best['validation_accuracy'],
                              'last_validation_accuracy': last['validation_accuracy'],
                              'reached_epoch_limit': len(history) == cfg['max_epochs']})
    for filename, rows in [('pilot_comparison.csv', pilot), ('training_checkpoint_review.csv', final)]:
        with (reports / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)
    for ax, name in zip(axes, NAMES):
        for condition, color in zip(cfg['conditions'], COLORS):
            folder = ROOT / 'outputs/runs' / f'{name}_{condition}_42'
            history = read(folder / 'history.json')
            selected = read(folder / 'validation.json')['best_epoch']
            epochs = [r['epoch'] for r in history]
            ax.plot(epochs, [r['train_loss'] for r in history], color=color, lw=1.6)
            ax.plot(epochs, [r['validation_loss'] for r in history], color=color, lw=1.6, ls='--')
            ax.plot(selected, history[selected - 1]['validation_loss'], 'o', color=color, ms=5)
        ax.set_title(NAMES[name], fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylim(0, 1.08)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Cross-entropy loss')
    handles = [Line2D([0], [0], color=COLORS[0], label='Intact'),
               Line2D([0], [0], color=COLORS[1], label='Permuted'),
               Line2D([0], [0], color='#333333', label='Training', ls='-'),
               Line2D([0], [0], color='#333333', label='Validation', ls='--'),
               Line2D([0], [0], color='#333333', label='Selected checkpoint', marker='o', ls='None')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=[0, .13, 1, 1])
    fig.savefig(figures / 'training_loss_walkthrough.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    for ax, name in zip(axes, NAMES):
        for lr, color in zip(cfg['learning_rates'][name], COLORS):
            folder = ROOT / 'outputs/pilot' / name / str(lr)
            history = read(folder / 'history.json')
            selected = read(folder / 'validation.json')['best_epoch']
            ax.plot([r['epoch'] for r in history], [100*r['validation_accuracy'] for r in history],
                    color=color, lw=1.5, label=f'lr = {lr}')
            ax.plot(selected, 100*history[selected - 1]['validation_accuracy'], 'o', color=color, ms=5)
        ax.set_title(NAMES[name], fontsize=14)
        ax.set_xlabel('Epoch')
        ax.legend(frameon=False, fontsize=10.5, loc='lower right')
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Validation accuracy (%)')
    fig.tight_layout()
    fig.savefig(figures / 'pilot_learning_rate_review.png', dpi=200)
    plt.close(fig)
    result = {'analysis_only': True, 'new_model_fits': 0, 'protocol_version': 2,
              'illustrated_seed': 42, 'pilot_fits_reviewed': len(pilot), 'final_histories_reviewed': len(final),
              'final_trainable_runs_stopped_before_limit': sum(not r['reached_epoch_limit'] for r in final),
              'selected_learning_rates': selection['selected_learning_rates'], 'pilots': pilot,
              'seed42_checkpoint_review': [r for r in final if r['seed'] == 42], 'source_sha256': sources,
              'training_loss_caveat': 'Training loss is the sample-weighted average of batch losses during updates. Validation loss uses a fixed end-of-epoch model. Their difference is not a matched-checkpoint generalisation-gap estimate.'}
    (reports / 'training_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['pilots', 'seed42_checkpoint_review', 'source_sha256']}, indent=2))

if __name__ == '__main__':
    main()
