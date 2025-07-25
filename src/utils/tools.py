import torch
import numpy as np
import os
import pickle

def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1
            
            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors

def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text

def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1
    
    return mask.bool()

class ClipProcessor:
    def __init__(self, model=None, vis_processor=None, text_processor=None, normal=False, device=None):
        self.model = model
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.normal= normal
        self.device= device
        self.caption_dict = {
            'abuse': 'a person is being abused or mistreated in a violent situation',
            'arrest': 'law enforcement officers are arresting a person',
            'arson': 'someone is intentionally setting fire to property',
            'assault': 'an individual is attacking or physically assaulting another person',
            'burglary': 'a break-in is occurring where someone is stealing from a property',
            'explosion': 'an explosion or blast is happening, causing destruction',
            'fighting': 'a fight is taking place between two or more people',
            'roadAccidents': 'a traffic accident is occurring on the road involving vehicles',
            'robbery': 'a person is committing a robbery, stealing forcefully from another person',
            'shooting': 'a person is firing a gun or there is a gunfight',
            'shoplifting': 'someone is stealthily stealing items from a store',
            'stealing': 'a person is stealing property or goods unlawfully',
            'testing_normal_videos_anomaly': 'a normal daily life activity is happening with no anomaly',
            'training_normal_videos_anomaly': 'a normal activity is being recorded with no unusual event',
            'vandalism': 'a person is damaging or defacing public or private property'
}
    def random_extract(self, feat, t_max):
        r = np.random.randint(feat.shape[0] - t_max)
        return feat[r : r+t_max, :]

    def text_grounded(self, clip_feat, snippets, label, video_id):
        """
        clip_feat: (B, D)
        label: class name
        video_id: unique identifier for the video (e.g., filename or path)
        """
        cache_path = f"./snippet_cache/{video_id}_top{snippets}.pkl"
        os.makedirs("./snippet_cache", exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                top_snippet_indices = pickle.load(f)
            return clip_feat[top_snippet_indices]

        # Otherwise compute via BLIP2ITM
        B = clip_feat.shape[0]
        caption = self.caption_dict[label]
        batch_size = min(100, B)
        idx = 0
        all_scores = []

        while idx + batch_size <= B:
            caption_list = [caption] * batch_size
            clip_feat_list = torch.from_numpy(clip_feat[idx:idx + batch_size]).float().to(self.device)
            itm_output = self.model({"image": clip_feat_list, "text_input": caption_list}, match_head="itm", precomputed_features=True)
            print("ITM Output: ", itm_output.shape)
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
            print("itm score: ", itm_scores.shape)

            for i in range(batch_size):
                score = itm_scores[i, 1].item()
                global_index = idx + i
                all_scores.append((score, global_index))
            idx += batch_size

        all_scores.sort(reverse=True, key=lambda x: x[0])
        top_snippets = all_scores[:snippets]
        top_snippets.sort(key=lambda x: x[1])
        top_snippet_indices = [index for (score, index) in top_snippets]

        # Save to cache
        with open(cache_path, "wb") as f:
            pickle.dump(top_snippet_indices, f)

        return clip_feat[top_snippet_indices]
    
    
    def uniform_extract(self, feat, t_max, avg=True):
        new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
        r = np.linspace(0, len(feat), t_max+1, dtype=np.int32)
        if avg == True:
            for i in range(t_max):
                if r[i]!=r[i+1]:
                    new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i,:] = feat[r[i],:]
        else:
            r = np.linspace(0, feat.shape[0]-1, t_max, dtype=np.uint16)
            new_feat = feat[r, :]

        return new_feat

    def pad(self, feat, min_len):
        clip_length = feat.shape[0]
        if clip_length <= min_len:
            return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
        else:
            return feat

    def process_feat(self, feat, snippets, clip_label, selection_method, filename):
        clip_length = feat.shape[0]
        if clip_length > snippets:
            if selection_method == 'random':
                return self.random_extract(feat, snippets), snippets
            elif selection_method == 'uniform':
                return self.uniform_extract(feat, snippets), snippets
            else:
                if self.normal:
                    return self.uniform_extract(feat, snippets), snippets
                else:
                    return self.text_grounded(feat, snippets, clip_label, filename), snippets
        else:
            return self.pad(feat, snippets), clip_length


    def process_split(self, feat, snippets):
        clip_length = feat.shape[0]
        if clip_length < snippets:
            return self.pad(feat, snippets), clip_length
        else:
            split_num = int(clip_length / snippets) + 1
            for i in range(split_num):
                if i == 0:
                    split_feat = feat[i*snippets:i*snippets+snippets, :].reshape(1, snippets, feat.shape[1])
                elif i < split_num - 1:
                    split_feat = np.concatenate([split_feat, feat[i*snippets:i*snippets+snippets, :].reshape(1, snippets, feat.shape[1])], axis=0)
                else:
                    split_feat = np.concatenate([split_feat, self.pad(feat[i*snippets:i*snippets+snippets, :], snippets).reshape(1, snippets, feat.shape[1])], axis=0)

            return split_feat, clip_length