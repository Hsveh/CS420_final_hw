"""
    child.py
    combine parent model and child model together
"""
import common
import numpy as np

"""
    threshold = max(confidence) - second max(confidence)
    common_path = common path of input data
    fig = image size
    model_path = model path
    parent_model = parent model version
    parent_model_iter = iteration
"""
threshold = 15
common_path = "../mnist"
fig = 28
model_path = 'model/'
parent_model = '1.4.0'
parent_model_iter = 60000

logger = common.create_logger('child', log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
data = common.Data(common_path+"/mnist_train/train_data.npy", common_path+"/mnist_train/mnist_train_label",
                   common_path+"/mnist_test/test_data.npy", common_path+"/mnist_test/mnist_test_label", 1, fig)

# reference from parent model
res = common.predict(model_path+parent_model, parent_model_iter, data.test_x, fig)

conf = []
conf_label = []
candidate = []
candidate_label = []
err = 0

for i in range(len(res)):
    m = np.max(res[i][0][0][0])
    temp = []
    for j in res[i][0][0][0]:
        if j < m:
            temp.append(j)
    m_ = np.max(temp)

    if m - m_ <= threshold:
        candidate.append(data.test_x[i])
        candidate_label.append(data.test_y_no_one_hot[i])
    else:
        conf.append(res[i])
        conf_label.append(data.test_y_no_one_hot[i])
logger.info("low confidence number: {}".format(len(candidate_label)))
infer_res, infer_conf, gt_conf = common.compare_result(conf, conf_label)

for i in range(len(infer_res)):
    if infer_res[i] != conf_label[i]:
        err += 1

res_ = common.predict('model/child0', 115000, candidate, 28)
infer_res_, infer_conf, gt_conf = common.compare_result(res_, candidate_label)

for i in range(len(infer_res_)):
    if infer_res_[i] != candidate_label[i]:
        err += 1
logger.info("Accuracy: {}".format(1.0-((err*1.0)/data.size_test)))
