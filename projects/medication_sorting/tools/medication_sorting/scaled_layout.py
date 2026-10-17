"""生成仅缩小两个药瓶的独立 Layout，并保存来源清单。"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from tools.medication_sorting.task_specs import TARGETS

BOTTLES = tuple(name for name in TARGETS if 'bottle' in name)
CAPTURE_LAYOUT = Path(__file__).resolve().parents[2]/'data/medication_sorting/scenes/drag_classfication_capture.json'


def make_scaled_bottle_layout(source, *, ratio=.7):
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('缩小比例必须在0–1之间且大于零')
    result = copy.deepcopy(source)
    found = {name: [] for name in BOTTLES}

    def walk(actors):
        for actor in actors:
            if actor['name'] in found:
                found[actor['name']].append(actor)
            walk(actor.get('children', []))

    walk(result['actors'])
    for name, actors in found.items():
        if len(actors) != 1:
            raise ValueError(f'必须包含唯一药瓶：{name}')
        actor = actors[0]
        old = actor['transform']['scale']
        if not isinstance(old, (int, float)) or not math.isfinite(old) or old <= 0:
            raise ValueError(f'药瓶原缩放比例无效：{name}')
        actor['transform']['scale'] = old * ratio
        for override in actor.get('entity_overrides') or []:
            if not override['entity_path']:
                for group in override['group_overrids']:
                    for prop in group['property_overrides']:
                        if prop['name'] == 'uniformScale':
                            prop['value'] = old * ratio
    return result


def resolve_experiment_layout(layout, *, profile='original'):
    if layout is None:
        if profile != 'original':
            raise ValueError('缩小药瓶配置必须显式指定 --layout，不能默认记录原尺寸场景')
        return CAPTURE_LAYOUT
    path = Path(layout)
    if profile in ('scaled_bottles_70', 'randomized_tabletop_70'):
        expected = make_scaled_bottle_layout(json.loads(CAPTURE_LAYOUT.read_text()), ratio=.7)
        if json.loads(path.read_text()) != expected:
            raise ValueError('场景与已定义的70%药瓶实验不一致')
    return path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'data/medication_sorting/scenes/drag_classfication_capture.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ratio', type=float, default=.7)
    args = parser.parse_args()
    contents = args.source.read_bytes()
    result = make_scaled_bottle_layout(json.loads(contents), ratio=args.ratio)
    manifest_path = args.output.with_suffix('.manifest.json')
    if args.output.resolve() == args.source.resolve() or args.output.exists() or manifest_path.exists():
        parser.error('不得覆盖源场景或已有实验文件')
    payload = (json.dumps(result, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    manifest = {'source': str(args.source.resolve()), 'source_sha256': hashlib.sha256(contents).hexdigest(),
                'output': str(args.output.resolve()), 'output_sha256': hashlib.sha256(payload).hexdigest(),
                'bottle_relative_scale': args.ratio, 'modified_actors': list(BOTTLES),
                'original_size_task_solved': False}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
