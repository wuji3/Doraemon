import glob
from utils.plots import Annotator
from functools import reduce
import platform
import shutil
import os
import torch.nn.functional as F
import cv2
from typing import Optional, Union
from dataset.basedataset import ImageDatasets
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

class Visualizer:
    
    @staticmethod
    def predict_images(model, 
                       dataloader, 
                       root, 
                       device, 
                       visual_path, 
                       class_indices: dict, 
                       logger, 
                       thresh: Union[float, list[float]], 
                       remove_label: bool, 
                       badcase: bool, 
                       is_cam: bool, 
                       target_class: Optional[str] = None):

        os.makedirs(visual_path, exist_ok=True)
        is_single_label = isinstance(thresh, (int, float)) and thresh == 0
        
        # Determine classification head type and activation function once
        class_head = 'ce' if is_single_label else 'bce'
        activation_fn = partial(F.softmax, dim=0) if class_head == 'ce' else partial(F.sigmoid)

        # Get target class index for multi-label case
        target_idx = None
        if not is_single_label:
            if isinstance(thresh, list):
                # Find target class index and its threshold
                for idx, class_name in class_indices.items():
                    if class_name == target_class:
                        target_idx = idx
                        break
            if target_idx is None:
                raise ValueError(f"Target class {target_class} not found in class indices")
                
            # Get and validate threshold for target class
            target_thresh = thresh[target_idx]
            if not isinstance(target_thresh, float):
                raise ValueError(f"Invalid threshold type for target class: {type(target_thresh)}. Must be float")
                
            # Update threshold to use only the target class threshold
            thresh = target_thresh
                
        # eval mode
        model.eval()
        n = len(dataloader)

        # cam
        if is_cam:
            from utils.cam import ClassActivationMaper
            cam = ClassActivationMaper(model, method='gradcam', device=device, transforms=dataloader.dataset.transforms)

        image_postfix_table = dict() # use for badcase
        for i, (img, inputs, img_path) in enumerate(dataloader):
            img = img[0]
            img_path = img_path[0]

            if not remove_label and is_cam:
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

            text = '\n'.join(f'{probs[j].item():.2f} {class_indices[j]}' for j in top5i)

            if not remove_label:
                annotator.text((32, 32), text, txt_color=(0, 0, 0))

            if remove_label or badcase:  # Write to file
                os.makedirs(os.path.join(visual_path, 'labels'), exist_ok=True)
                image_postfix_table[os.path.basename(os.path.splitext(img_path)[0] + '.txt')] = os.path.splitext(img_path)[1]
                with open(os.path.join(visual_path, 'labels', os.path.basename(os.path.splitext(img_path)[0] + '.txt')), 'a') as f:
                    f.write(text + '\n')

            logger.console(f"[{i+1}|{n}] " + os.path.basename(img_path) +" " + reduce(lambda x,y: x + " "+ y, text.split()))

            if not remove_label and is_cam:
                img = np.hstack([cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cam_image])
                cv2.imwrite(os.path.join(visual_path, os.path.basename(img_path)), img)
            else: img.save(os.path.join(visual_path, os.path.basename(img_path)))

        if badcase:
            os.makedirs(os.path.join(visual_path, 'bad_case'), exist_ok=True)
            for txt in glob.glob(os.path.join(visual_path, 'labels', '*.txt')):
                with open(txt, 'r') as f:
                    lines = f.readlines()
                    if is_single_label:
                        # Single-label case: check if top prediction matches target class
                        is_badcase = lines[0].split()[1] != target_class
                    else:
                        # Multi-label case: check if target class probability exceeds threshold
                        is_badcase = True
                        for line in lines:
                            prob, class_name = float(line.split()[0]), line.split()[1]
                            if class_name == target_class and prob >= thresh:
                                is_badcase = False
                                break
                
                if is_badcase:
                    try:
                        shutil.move(
                            os.path.join(visual_path, 
                                       os.path.basename(txt).replace('.txt', 
                                       image_postfix_table[os.path.basename(txt)])), 
                            os.path.dirname(txt).replace('labels','bad_case')
                        )
                    except FileNotFoundError:
                        print(f'FileNotFoundError->{txt}')

    @staticmethod
    def visualize_results(query, 
                          retrieval_results, 
                          scores, 
                          ground_truths, 
                          savedir,
                          max_rank=5,
                          ):

        os.makedirs(savedir, exist_ok=True)

        fig, axes = plt.subplots(2, max_rank + 1, figsize=(3 * (max_rank + 1), 12))

        for ax in axes.ravel():
            ax.set_axis_off()
        # Display the query image in the first position of the second row
        query_img = ImageDatasets.read_image(query)
        ax = fig.add_subplot(2, max_rank + 1, max_rank + 2)
        ax.imshow(query_img)
        ax.set_title('Query')
        ax.axis("off")

        # Display the ground truth images
        for i in range(min(5, len(ground_truths))):
            gt_img = ImageDatasets.read_image(ground_truths[i])
            ax = fig.add_subplot(2, max_rank + 1, i + 1)
            ax.imshow(gt_img)
            ax.set_title('Ground Truth')
            ax.axis("off")

        # Display the retrieval images
        for i in range(max_rank):
            retrieval_img = ImageDatasets.read_image(retrieval_results[i])

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