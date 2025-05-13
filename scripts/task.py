import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.all import *
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from matplotlib.font_manager import FontProperties


# def splitter(path):
#     fnames = get_image_files(path)
#     n_splits = 2
#     folds = [[] for _ in range(n_splits)]
#     names = [Path(fname).name for fname in fnames]
#     for i, fname in enumerate(fnames):
#         folds[i % n_splits].append(fname)
#     return folds

def carregar_e_processar_dados(caminho_arquivo, mode):
    path = Path(caminho_arquivo)
    print(f'Images from: {str(path)}')
    if mode == 'train':
        # train_fnames = splitter(path/"Train")
        train_fnames = get_image_files(path/"Train")

        # print(f"Fold ID: {fold_id}")
        augs = [RandomResizedCropGPU(size=224, min_scale=0.75), Rotate(), Zoom()]
        dblock = DataBlock(blocks=(ImageBlock(cls=PILImage), CategoryBlock),
                            splitter=RandomSplitter(valid_pct=0.2, seed=23),
                            get_y=parent_label,
                            item_tfms=Resize(512, method="squish"),
                            batch_tfms=augs,
                            )

        print("Creating dataloaders...")
        dls = dblock.dataloaders(train_fnames, num_workers=0)
        print("Dataloaders created successfully.")
        print(dls.c, len(dls.train_ds), len(dls.valid_ds))
    elif mode == 'test':
        all_files= get_image_files(path)

        augs = [RandomResizedCropGPU(size=224, min_scale=0.75), Rotate(), Zoom()]
        dblock = DataBlock(blocks=(ImageBlock(cls=PILImage), CategoryBlock),
                            splitter=GrandparentSplitter(train_name='Train', valid_name='Test'),
                            get_y=parent_label,
                            item_tfms=Resize(512, method="squish"),
                            batch_tfms=augs,
                            )
        print("Creating dataloaders...")
        dls = dblock.dataloaders(all_files, num_workers=0)
        print("Dataloaders created successfully.")
    return dls  

def treinar_modelo_flower(model, dls, foldID):
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    
    if foldID == 0:
        learn.save('FL-DatasetYildirim-FL')
        print('Salvando FL-DatasetYildirim-FL')
    elif foldID == 1:
        learn.save('FL-DatasetGoogleDrive-FL')
        print('Salvando FL-DatasetGoogleDrive-FL')

    learn.fit_one_cycle(10, 1e-2)

    
    return learn



def avaliar_modelo_flower(model, dls):
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    val_loss, acc = learn.validate()

    interp = ClassificationInterpretation.from_learner(learn)

    preds, targets = learn.get_preds(dl=dls.valid)
    predicted_labels = np.argmax(preds, axis=1)
    true_labels = targets

    report = classification_report(true_labels, predicted_labels, zero_division=0, target_names=dls.vocab)
    print(report)

    interp.plot_confusion_matrix()
    
    predicted_labels = predicted_labels.tolist()
    true_labels = true_labels.tolist()

    preds = preds[:, 1]
    preds = preds.tolist()

    return [val_loss, acc, predicted_labels, true_labels, preds]

def plotar_graficos(cm, fpr, tpr, roc_auc, loss_history, acc_history):
    pf = FontProperties(fname="TimesNewNormalRegular.ttf")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axs.flatten()

    sns.heatmap(cm.astype(int), annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("Matriz de Confusão")
    ax1.set_xlabel("Previsões")
    ax1.set_ylabel("Valores Reais")

    # ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"Curva ROC (AUC = {roc_auc:.2f})")
    # ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # ax2.set_xlim()
    # ax2.set_ylim()
    # ax2.set_xlabel("Taxa de Falsos Positivos")
    # ax2.set_ylabel("Taxa de Verdadeiros Positivos")
    # ax2.set_title("Curva ROC")
    # ax2.legend(loc="lower right")

    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score_value:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.xlim([0.0, 1.0])
    ax2.ylim([0.0, 1.05])
    ax2.xlabel('True negative rate (FPR)', fontproperties=pf)
    ax2.ylabel('True positive rate (TPR)', fontproperties=pf)
    ax2.title("ROC Curve", fontproperties=pf)
    ax2.legend(loc="lower right", prop=pf)
    ax2.grid(True)
    ax2.savefig('grafico_curvas.pdf')

    rounds = list(range(len(loss_history)))
    ax3.plot(rounds, loss_history, 'b-', label='Loss')
    ax3.set_title("Validation Loss", fontproperties=pf)
    ax3.set_xlabel("Round", fontproperties=pf)
    ax3.set_ylabel("Loss", fontproperties=pf)
    ax3.grid(True)
    ax3.legend(prop=pf)
    ax3.savefig('grafico_curvasH.pdf')

    rounds_acc = list(range(len(acc_history)))
    ax4.plot(rounds_acc, acc_history, 'g-', label='Acurácia')
    ax4.set_title("Histórico de Acurácia")
    ax4.set_xlabel("Round")
    ax4.set_ylabel("Acurácia")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()