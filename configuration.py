pose_home = 'pose/ViTPose'
pose_env = 'vitpose'
alpha_pose_env = 'alpha_pose'
media_pose_env = 'mediapose'

pose_detection_pipeline = 'vitpose'
openpose_bin_dir = './bin/OpenPoseDemo.exe'
openpose_use_cache = True
number_recognition_pipeline = 'str' #str or CNN

str_home = 'str/parseq/'
str_env = 'parseq2'
str_platform = 'cu113'

# centroids
reid_env = 'centroids'
reid_script = 'centroid_reid.py'

reid_home = 'reid/'


dataset = {'SoccerNet':
                {'root_dir': './data/SoccerNet',
                 'working_dir': './out/SoccerNetResults',
                 'test': {
                        'images': 'test/images',
                        'gt': 'test/test_gt.json',
                        'feature_output_folder': 'out/SoccerNetResults/test',
                        'illegible_result': 'illegible.json',
                        'soccer_ball_list': 'soccer_ball.json',
                        'sim_filtered': 'test/main_subject_0.4.json',
                        'gauss_filtered': 'test/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'legible.json',
                        'raw_legible_result': 'raw_legible_resnet34.json',
                        'pose_input_json': 'pose_input.json',
                        'pose_output_json': 'pose_results.json',
                        'crops_folder': 'crops',
                        'jersey_id_result': 'jersey_id_results.json',
                        'final_result': 'final_results.json'
                    },
                'test_sr': {
                        'images': 'test_sr/images',
                        'gt': 'test_sr/test_sr_gt.json',
                        'feature_output_folder': 'out/SoccerNetResults/test_sr',
                        'illegible_result': 'illegible_test_sr.json',
                        'soccer_ball_list': 'soccer_ball_test_sr.json',
                        'sim_filtered': 'test_sr/main_subject_0.4.json',
                        'gauss_filtered': 'test_sr/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'legible_test_sr.json',
                        'raw_legible_result': 'raw_legible_resnet34_test_sr.json',
                        'pose_input_json': 'pose_input_test_sr.json',
                        'pose_output_json': 'pose_results_test_sr.json',
                        'crops_folder': 'crops_test_sr',
                        'jersey_id_result': 'jersey_id_results_test_sr.json',
                        'final_result': 'final_results_test_sr.json'
                    },
                 'val': {
                        'images': 'val/images',
                        'gt': 'val/val_gt.json',
                        'feature_output_folder': 'out/SoccerNetResults/val',
                        'illegible_result': 'illegible_val.json',
                        'legible_result': 'legible_val.json',
                        'soccer_ball_list': 'soccer_ball_val.json',
                        'crops_folder': 'crops_val',
                        'sim_filtered': 'val/main_subject_0.4.json',
                        'gauss_filtered': 'val/main_subject_gauss_th=3.5_r=3.json',
                        'pose_input_json': 'pose_input_val.json',
                        'pose_output_json': 'pose_results_val.json',
                        'jersey_id_result': 'jersey_id_results_validation.json'
                    },
                 'val_sr': {
                     'images': 'val_sr/images',
                     'gt': 'val_sr/val_sr_gt.json',
                     'feature_output_folder': 'out/SoccerNetResults/val_sr',
                     'illegible_result': 'illegible_val_sr.json',
                     'legible_result': 'legible_val_sr.json',
                     'soccer_ball_list': 'soccer_ball_val_sr.json',
                     'crops_folder': 'crops_val_sr',
                     'sim_filtered': 'val_sr/main_subject_0.4.json',
                     'gauss_filtered': 'val_sr/main_subject_gauss_th=3.5_r=3.json',
                     'pose_input_json': 'pose_input_val_sr.json',
                     'pose_output_json': 'pose_results_val_sr.json',
                     'jersey_id_result': 'jersey_id_results_validation_sr.json'
                 },
                 'train': {
                     'images': 'train/images',
                     'gt': 'train/train_gt.json',
                     'feature_output_folder': 'out/SoccerNetResults/train',
                     'illegible_result': 'illegible_train.json',
                     'legible_result': 'legible_train.json',
                     'soccer_ball_list': 'soccer_ball_train.json',
                     'sim_filtered': 'train/main_subject_0.4.json',
                     'gauss_filtered': 'train/main_subject_gauss_th=3.5_r=3.json',
                     'pose_input_json': 'pose_input_train.json',
                     'pose_output_json': 'pose_results_train.json',
                     'raw_legible_result': 'train_raw_legible_combined.json'
                 },
                 'challenge': {
                        'images': 'challenge/images',
                        'feature_output_folder': 'out/SoccerNetResults/challenge',
                        'gt': '',
                        'illegible_result': 'challenge_illegible.json',
                        'soccer_ball_list': 'challenge_soccer_ball.json',
                        'sim_filtered': 'challenge/main_subject_0.4.json',
                        'gauss_filtered': 'challenge/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'challenge_legible.json',
                        'pose_input_json': 'challenge_pose_input.json',
                        'pose_output_json': 'challenge_pose_results.json',
                        'crops_folder': 'challenge_crops',
                        'jersey_id_result': 'challenge_jersey_id_results.json',
                        'final_result': 'challenge_final_results.json',
                        'raw_legible_result': 'challenge_raw_legible_vit.json'
                 },
                'demo': {
                        'images': 'demo/images',
                        'feature_output_folder': 'out/SoccerNetResults/demo',
                        'gt': '',
                        'illegible_result': 'demo_illegible.json',
                        'soccer_ball_list': 'demo_soccer_ball.json',
                        'sim_filtered': 'demo/main_subject_0.4.json',
                        'gauss_filtered': 'demo/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'demo_legible.json',
                        'pose_input_json': 'demo_pose_input.json',
                        'pose_output_json': 'demo_pose_results.json',
                        'crops_folder': 'demo_crops',
                        'jersey_id_result': 'demo_jersey_id_results.json',
                        'final_result': 'demo_final_results.json',
                        'raw_legible_result': 'demo_raw_legible_vit.json'
                 },
                'challenge_sr': {
                        'images': 'challenge_sr/images',
                        'gt': 'challenge_sr/challenge_sr_gt.json',
                        'feature_output_folder': 'out/SoccerNetResults/challenge_sr',
                        'illegible_result': 'illegible_challenge_sr.json',
                        'soccer_ball_list': 'soccer_ball_challenge_sr.json',
                        'sim_filtered': 'challenge_sr/main_subject_0.4.json',
                        'gauss_filtered': 'challenge_sr/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'legible_challenge_sr.json',
                        'raw_legible_result': 'raw_legible_resnet34_challenge_sr.json',
                        'pose_input_json': 'pose_input_challenge_sr.json',
                        'pose_output_json': 'pose_results_challenge_sr.json',
                        'crops_folder': 'crops_challenge_sr',
                        'jersey_id_result': 'jersey_id_results_challenge_sr.json',
                        'final_result': 'final_results_challenge_sr.json'
                    },
                 'numbers_data': 'lmdb',

                 'legibility_model': "models/legibility_resnet34_soccer_20240215.pth",
                 'legibility_model_arch': "resnet34",

                 'legibility_model_url':  "https://drive.google.com/uc?id=18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw",
                 'pose_model_url': 'https://drive.google.com/uc?id=1A3ftF118IcxMn_QONndR-8dPWpf7XzdV',
                 'str_model': 'models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt',
                 'cnn_model': 'models/cnn_numrecognition_colab.pth',

                 #'str_model': 'pretrained=parseq',
                 'str_model_url': "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
                },
           "Hockey": {
                 'root_dir': 'data/Hockey',
                 'legibility_data': 'legibility_dataset',
                 'numbers_data': 'jersey_number_dataset/jersey_numbers_lmdb',
                 'legibility_model':  'models/legibility_resnet34_hockey_20240201.pth',
                 'legibility_model_url':  "https://drive.google.com/uc?id=1RfxINtZ_wCNVF8iZsiMYuFOP7KMgqgDp",
                 'str_model': 'models/parseq_epoch=3-step=95-val_accuracy=98.7903-val_NED=99.3952.ckpt',
                 'str_model_url': "https://drive.google.com/uc?id=1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE",
            }
        }