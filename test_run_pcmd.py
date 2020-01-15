import os
import glob 
import concurrent.futures as cf
from config import window_size
#filelist = [
#'3_61921-.csv','3_62112-.csv','3_62247-.csv', '3_62249-.csv','3_62263-.csv','3_62287-.csv','3_62921-.csv','3_64791-.csv','3_66247-.csv','3_66669-.csv','3_66670-.csv',
#'1_660674.csv','1_661703.csv','1_661814.csv','1_663429.csv','1_663768.csv', '1_664101.csv','1_665169.csv','1_669067.csv','1_669136.csv',
#'4_732142.csv',  '4_733064.csv',  '4_733108.csv',  '4_733304.csv', '4_733342.csv',  '4_733726.csv',  '4_733793.csv',  '4_733942.csv',
#'4_733026.csv',  '4_733093.csv',  '4_733200.csv',  '4_733341.csv',  '4_733708.csv',  '4_733764.csv', '4_733900.csv', '4_733944.csv'
#'2_520154.csv', '2_520691.csv', '2_521252.csv', '2_521482.csv','2_521532.csv', '2_523962.csv', '2_525275.csv',
#'2_520688.csv',  '2_521125.csv', '2_521325.csv',  '2_521530.csv' , '2_521535.csv' , '2_525246.csv',  '2_525549.csv',  
#'1_660149.csv','1_660720.csv','1_661752.csv','1_661851.csv','1_663515.csv','1_664054.csv','1_664893.csv','1_665429.csv','1_669071.csv' , 
#]

#filelist = ['3_66670-.csv']
def worker(f):
    cutf = f.split('.csv')[0] + '_' + str(window_size)
    strcmd = "python main.py --filename {} --epoch 150 --model_output ./model_train/output_{} --preprocess 1 --serving_output_dir ./model_serve/model_{} --transform_dir ../tft_output/tft_train_data_{} --train_raw_data pcmd_data_full/ --batch_size 1024 --prediction_dir prediction_output ../data/ > logs/out2_{} 2> logs/error2_{} ".format(
        f,
        cutf,
        cutf,
        cutf,
        cutf,
        cutf,
        cutf)
    print (strcmd)
    return os.system(strcmd)
#    return 0
def test_run():
     futures = []
     with cf.ProcessPoolExecutor(9) as pp:
         for f in glob.glob("../pcmd_data_full_backup/*WAL*.csv"):
                 f1 = os.path.basename(f)
                 os.system(" cp ../pcmd_data_full_backup/{}  pcmd_data_full/".format(f1) )
                 if 'test' in f1:
                    continue
                 futures.append(pp.submit(worker, f1))
     for future in cf.as_completed(futures):
         print (future.result())
if __name__ == "__main__":
    # os.system(" nohup nvidia-smi -q --display=UTILIZATION -l 5  -a > nvidia_out &")
     test_run()
     #os.system("pkill nvidia-smi")

