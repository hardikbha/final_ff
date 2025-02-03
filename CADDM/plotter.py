import cv2
import os

def test():
    args = args_func()

    # Load configs
    cfg = load_config(args.cfg)

    # Initialize model
    net = model.get(backbone=cfg['model']['backbone'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net = nn.DataParallel(net)
    net.eval()
    if cfg['model']['ckpt']:
        net = load_checkpoint(cfg['model']['ckpt'], net, device)

    # Load testing data
    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
    test_dataset = DeepfakeDataset('test', cfg)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['test']['batch_size'],
                             shuffle=False, num_workers=4)

    # Directory to save processed frames
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Video writer for saving the video
    video_path = "output_video.mp4"
    video_writer = None

    frame_pred_list = list()
    frame_label_list = list()
    video_name_list = list()

    for batch_data, batch_labels in test_loader:

        labels, video_name = batch_labels
        labels = labels.long()

        outputs = net(batch_data)
        outputs = outputs[:, 1]  # Assuming binary classification (Fake/Real)
        predictions = (outputs > 0.5).int()  # Binary predictions: 1 for Fake, 0 for Real

        frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
        frame_label_list.extend(labels.detach().cpu().numpy().tolist())
        video_name_list.extend(list(video_name))

        # Process frames for visualization
        for i, frame in enumerate(batch_data):
            frame = frame.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy image
            frame = (frame * 255).astype('uint8')  # Scale pixel values
            pred = predictions[i].item()
            label_text = "Fake" if pred == 1 else "Real"
            color = (0, 0, 255) if pred == 1 else (0, 255, 0)  # Red for Fake, Green for Real

            # Draw label and box
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            height, width, _ = frame.shape
            cv2.rectangle(frame, (10, 10), (width - 10, height - 10), color, 3)  # Draw frame border

            # Save frame as an image
            frame_path = os.path.join(output_dir, f"{video_name[i]}_{i}.jpg")
            cv2.imwrite(frame_path, frame)

            # Initialize video writer
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

            # Write frame to video
            video_writer.write(frame)

    # Release video writer
    if video_writer is not None:
        video_writer.release()

    # Compute metrics
    f_auc = roc_auc_score(frame_label_list, frame_pred_list)
    v_auc = get_video_auc(frame_label_list, video_name_list, frame_pred_list)
    print(f"Frame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    print(f"Video-AUC of {cfg['dataset']['name']} is {v_auc:.4f}")
    print(f"Processed video saved at {video_path}")