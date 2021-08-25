from argparse import ArgumentParser
import glob
import os

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--source-path', default='/home/ubuntu/data/test_images/',
        help='Input images path'
    )
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--target-path', default='/home/ubuntu/inference_results',
        help='Path for the output images.'
    )
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.CLASSES = ['logo'] * 50000

    for img_path in glob.glob(os.path.join(args.source_path, '*')):
        print(img_path)
        result = inference_detector(model, img_path)
        print(result)
        # show the results
        model.show_result(
            img_path,
            result,
            score_thr=0,
            bbox_color=(72, 101, 241),
            text_color=(72, 101, 241),
            out_file=os.path.join(args.target_path, img_path.split('/')[-1])
        )


if __name__ == '__main__':
    main()
