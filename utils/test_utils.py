
import torch 

STATE_DIC_PATH='temp_data/saved/someOutput/latest/model.pth'

def generate_prediction_file():
    prediction=gen_prediction()
    turn_id=gen_turn_id()
    paragraph_id=gen_paragraph_id()
    save_to_json(prediction,turn_id,paragraph_id)


def load_model():
    save_dic=torch.load(STATE_DIC_PATH)
    model=Model(config, MODELS[config['model_name']], self.device, tokenizer).to(self.device)
    model.load_state_dict(save_dic['model'])
    model.eval()













if __name__=='__main__':
    generate_prediction_file()