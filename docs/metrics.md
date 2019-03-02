# Available Metrics

Semantic Segmentation IoU
-------------------------

### calc_semantic_segmentation_confusion
```
ncc.metrics.segmentation_metrics.calc_semantic_segmentation_confusion(pred_labels, gt_labels)
```

### calc_semantic_segmentation_iou
```
ncc.metrics.segmentation_metrics.calc_semantic_segmentation_iou(pred_labels, gt_labels)
```

### eval_semantic_segmentation
```
ncc.metrics.segmentation_metrics.eval_semantic_segmentation(pred_labels, gt_labels)
```

### detection_rate_confusions
```
ncc.metrics.segmentation_metrics.detection_rate_confusions(pred_labels, gt_labels, nb_classes)
```

### plot_confusion_matrix
```
ncc.metrics.segmentation_metrics.plot_confusion_matrix(cm, classes,
                                                       normalize=False,
                                                       title='Confusion matrix',
                                                       cmap=plt.cm.Blues,
                                                       save_file=None)
```