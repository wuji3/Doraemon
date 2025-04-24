import glob
from ...utils.plots import Annotator
import platform
import shutil
import os
import torch.nn.functional as F
import cv2
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

class Visualizer:
    
    @staticmethod
    def predict_images(model, 
                       dataloader, 
                       device, 
                       visual_path, 
                       class_indices: dict, 
                       logger, 
                       thresh: Union[float, list[float]], 
                       infer_option: str = 'default',
                       ):
        """
        Args:
            infer_option: 
                - 'default': Infer + Visualize + GradCAM + Badcase
                - 'autolabel': Infer + Label
        """
        os.makedirs(visual_path, exist_ok=True)
        is_single_label = isinstance(thresh, (int, float)) and thresh == 0
        
        # Determine classification head type and activation function once
        class_head = 'ce' if is_single_label else 'bce'
        activation_fn = partial(F.softmax, dim=0) if class_head == 'ce' else partial(F.sigmoid)

        target_classes = dataloader.dataset.classes if dataloader.dataset.classes is not None else None
        if target_classes and not isinstance(target_classes, list):
            target_classes = [target_classes]
            
        # 获取每个目标类别的索引和阈值
        target_indices = []
        target_thresholds = []
        if not is_single_label and isinstance(thresh, list):
            for target_class in target_classes:
                target_idx = None
                for idx, class_name in class_indices.items():
                    if class_name == target_class:
                        target_idx = idx
                        target_indices.append(idx)
                        break
                if target_idx is None:
                    raise ValueError(f"Target class {target_class} not found in class indices")
                
                # 获取并验证目标类别的阈值
                target_thresh = thresh[target_idx]
                if not isinstance(target_thresh, float):
                    raise ValueError(f"Invalid threshold type for target class: {type(target_thresh)}. Must be float")
                target_thresholds.append(target_thresh)
                
        # Initialize CAM if in default mode
        if infer_option == 'default':
            from ...utils.cam import ClassActivationMaper
            cam = ClassActivationMaper(model, method='gradcam', device=device, transforms=dataloader.dataset.transforms)

        # eval mode
        model.eval()
        n = len(dataloader)

        fixed_class_length = 15
        progress_width = len(str(n))

        image_postfix_table = dict()
        for i, (img, inputs, img_path, gt_labels) in enumerate(dataloader):
            img = img[0]
            img_path = img_path[0]
            gt_label = gt_labels[0] if gt_labels is not None else None

            if infer_option == 'default':
                cam_image = cam(image=img, input_tensor=inputs, dsize=img.size)
                cam_image = cv2.resize(cam_image, img.size, interpolation=cv2.INTER_LINEAR)

            # system
            if platform.system().lower() == 'windows':
                annotator = Annotator(img, font=r'C:/WINDOWS/FONTS/SIMSUN.TTC') # windows
            else:
                annotator = Annotator(img) # linux

            # transforms
            inputs = inputs.to(device)
            # forward
            logits = model(inputs).squeeze()

            # post process using pre-determined activation function
            probs = activation_fn(logits)
            top5i = probs.argsort(0, descending=True)[:5].tolist()
            
            text = '\n'.join(f'{class_indices[j]:<{fixed_class_length}} {probs[j].item():.2f}' for j in top5i)
            
            formatted_predictions = '      '.join(f'{class_indices[j]:<{fixed_class_length}}{probs[j].item():.2f}' for j in top5i)
            logger.console(f"[{i+1:>{progress_width}}|{n:<{progress_width}}] {os.path.basename(img_path):<20} {formatted_predictions}")

            annotator.text((32, 32), text, txt_color=(0, 0, 0))

            # Save predictions and ground truth
            save_dir = os.path.join(visual_path, 'labels')
            os.makedirs(save_dir, exist_ok=True)
            image_postfix_table[os.path.basename(os.path.splitext(img_path)[0] + '.txt')] = {
                'ext': os.path.splitext(img_path)[1],
                'gt': gt_label
            }
            with open(os.path.join(save_dir, os.path.basename(os.path.splitext(img_path)[0] + '.txt')), 'a') as f:
                f.write(text + '\n')

            if infer_option == 'default':
                img = np.hstack([cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cam_image])
                cv2.imwrite(os.path.join(visual_path, os.path.basename(img_path)), img)

        # Process badcases only in default mode
        if infer_option == 'default':
            badcase_root = os.path.join(visual_path, 'badcase')
            if target_classes:
                for target_class in target_classes:
                    os.makedirs(os.path.join(badcase_root, target_class), exist_ok=True)
            else:
                os.makedirs(badcase_root, exist_ok=True)

            for txt in glob.glob(os.path.join(visual_path, 'labels', '*.txt')):
                with open(txt, 'r') as f:
                    lines = f.readlines()
                    gt = image_postfix_table[os.path.basename(txt)]['gt']
                    
                    if gt is None:
                        continue
                    
                    if is_single_label:
                        pred_class = lines[0].rsplit(' ',1)[0]
                        is_badcase = pred_class != gt
                        target_class = gt  # For determining save path
                    else:
                        # Multi-label case, check each target class
                        is_badcase = False
                        target_class = None  # For determining save path
                        for target, thresh in zip(target_classes, target_thresholds):
                            found_correct_pred = False
                            for line in lines:
                                class_name, prob = line.rsplit(' ', 1)
                                prob = float(prob)
                                if class_name == target:
                                    if prob < thresh:
                                        is_badcase = True
                                        target_class = target  # Record the class causing the badcase
                                    found_correct_pred = True
                                    break
                            if not found_correct_pred:
                                is_badcase = True
                                target_class = target
                
                if is_badcase:
                    try:
                        source_path = os.path.join(visual_path, 
                                                 os.path.basename(txt).replace('.txt', 
                                                 image_postfix_table[os.path.basename(txt)]['ext']))
                        if target_class and target_classes:
                            dest_path = os.path.join(badcase_root, target_class)
                        else:
                            dest_path = badcase_root
                        shutil.move(source_path, dest_path)
                    except FileNotFoundError:
                        print(f'FileNotFoundError->{txt}')

    @staticmethod
    def visualize_results(query, 
                          retrieval_results, 
                          scores, 
                          ground_truths, 
                          savedir,
                          max_rank=5,
                          query_dataset=None,
                          gallery_dataset=None
                          ):

        os.makedirs(savedir, exist_ok=True)

        fig, axes = plt.subplots(2, max_rank + 1, figsize=(3 * (max_rank + 1), 12))

        for ax in axes.ravel():
            ax.set_axis_off()
        # Display the query image in the first position of the second row
        query_img = query_dataset.get_image(query)
        ax = fig.add_subplot(2, max_rank + 1, max_rank + 2)
        ax.imshow(query_img)
        ax.set_title('Query')
        ax.axis("off")

        # Display the ground truth images
        for i in range(min(5, len(ground_truths))):
            gt_img = gallery_dataset.get_image(ground_truths[i])
            ax = fig.add_subplot(2, max_rank + 1, i + 1)
            ax.imshow(gt_img)
            ax.set_title('Ground Truth')
            ax.axis("off")

        # Display the retrieval images
        for i in range(max_rank):
            retrieval_img = gallery_dataset.get_image(retrieval_results[i])

            score = scores[i]
            is_tp = retrieval_results[i] in ground_truths
            label = 'true' if is_tp else 'false'
            color = (1, 0, 0)

            ax = fig.add_subplot(2, max_rank + 1, (max_rank + 1) + i + 2)
            if is_tp:
                ax.add_patch(plt.Rectangle(xy=(0, 0), width=retrieval_img.width - 1,
                                           height=retrieval_img.height - 1, edgecolor=color,
                                           fill=False, linewidth=8))
            ax.imshow(retrieval_img)
            ax.set_title('{:.4f}/{}'.format(score, label))
            ax.axis("off")

        #plt.tight_layout()
        image_id = os.path.basename(os.path.dirname(query))
        image_name = os.path.basename(query)
        image_unique = image_id + '_' + image_name
        fig.savefig(os.path.join(savedir, image_unique))
        plt.close(fig)