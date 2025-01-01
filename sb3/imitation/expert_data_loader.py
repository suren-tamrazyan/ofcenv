import json
import numpy as np
from imitation.data.types import TrajectoryWithRew

def load_expert_trajectories(json_path):
    """Загружает экспертные траектории из JSON файла."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    trajectories = []
    
    for episode in data['episodes']:
        observations = []
        actions = []
        rewards = []
        
        steps = episode['observations']
        for i in range(len(steps) - 1):  # Проходим до предпоследнего шага
            observations.append(np.array(steps[i]['state']))
            actions.append(steps[i]['action'])
            rewards.append(float(steps[i]['reward']))
        
        # Добавляем последнее наблюдение
        observations.append(np.array(steps[-1]['state']))
            
        # Преобразуем в numpy массивы
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        
        # Создаем траекторию
        trajectory = TrajectoryWithRew(
            obs=observations,  # На 1 больше чем actions
            acts=actions,
            infos=None,
            terminal=True,
            rews=rewards
        )
        
        trajectories.append(trajectory)
        
    return trajectories 