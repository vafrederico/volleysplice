#!/usr/bin/env python3
"""Real-frame numerical/gradient checks before a fold-local student study."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
import numpy as np
import torch
from analysis import neural_mobile_distillation as study
from analysis import dinov2_embeddings as dino
from analysis import mobile_visual_features as mobile
from analysis.distillation_image_inputs import ASSETS, COMMIT, WEIGHT_SHA
from analysis.neural_context_development import identity, read, write_immutable


def state_hash(encoder, projector):
    digest = hashlib.sha256()
    for module in (encoder, projector):
        for key, value in module.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    output = study.OUTPUT/'distillation-engineering-v1/report.json'
    if output.exists():
        raise FileExistsError(output)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Deterministic workspace required')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device='cuda'
    encoder, projector = study.build_student(3407, device)
    teacher = dino.load_pinned_dinov2(ASSETS/f'dinov2-{COMMIT}', repository_commit=COMMIT,
        checkpoint=ASSETS/'dinov2_vits14_pretrain.pth', checkpoint_sha256=WEIGHT_SHA, device=device)
    records=[]
    all_pixels=[]
    all_targets=[]
    for identifier in (private_value('grass-source-03'),private_value('grass-source-01'),private_value('indoor-source-01'),private_value('indoor-source-08')):
        receipt=read(study.IMAGES/identifier/'receipt.json')
        selected=np.linspace(0,receipt['teachingFrames']-1,8,dtype=np.int64)
        with np.load(study.verify_once(receipt['arrays']['timing']),allow_pickle=False) as timing:
            indexes=timing['teaching_indexes'][selected]
            boxes=timing['boxes'][indexes]
        pixels=np.load(study.verify_once(receipt['arrays']['images224']),mmap_mode='r')[indexes].copy()
        teacher_pixels=np.load(study.verify_once(receipt['arrays']['teaching336']),mmap_mode='r')[selected].copy()
        values=(pixels.astype(np.float32)/255.-mobile.RGB_MEAN[None,:,None,None])/mobile.RGB_STD[None,:,None,None]
        teacher_values=(teacher_pixels.astype(np.float32)/255.-dino.IMAGENET_RGB_MEAN[None,:,None,None])/dino.IMAGENET_RGB_STD[None,:,None,None]
        with torch.no_grad():
            first=encoder(study.normalize_pixels(pixels,device))
            reference=encoder(torch.from_numpy(values).to(device))
            torch.testing.assert_close(first,reference,atol=1e-5,rtol=1e-4)
            weights=torch.from_numpy(mobile.regional_pool_weights(boxes,*first.shape[-2:])).to(device)
            pooled=torch.einsum('bchw,brhw->brc',first,weights).cpu().numpy()
            target=dino._extract_feature_tokens(teacher.model,torch,torch.from_numpy(teacher_values).to(device),336).cpu().numpy()
        original=mobile.load_mobile_visual_cache(study.verify_once(receipt['contract']['originalMobileCache']))
        expected=original.tokens[indexes].astype(np.float16)
        # Two half-precision ULPs account for the historical cache's rounding;
        # fresh FP32 preprocessing equality is separately checked above.
        tolerance=2*np.abs(np.spacing(expected)).astype(np.float32)+1e-5
        difference=np.abs(pooled-expected.astype(np.float32))
        study.require(np.all(difference<=tolerance),'Unchanged encoder does not reproduce historical mobile tokens')
        records.append({'id':identifier,'frameIndexes':indexes.tolist(),'inputReceipt':identity(study.IMAGES/identifier/'receipt.json'),
                        'maximumSpatialDifference':float((first-reference).abs().max().cpu()),
                        'maximumHistoricalTokenDifference':float(difference.max()),'historicalHalfUlpTolerancePassed':True})
        all_pixels.append(pixels)
        all_targets.append(target)
    del teacher,encoder,projector
    torch.cuda.empty_cache()
    pixels,targets=np.concatenate(all_pixels),np.concatenate(all_targets)
    runs=[]
    for repetition in range(2):
        encoder,projector=study.build_student(3407,device)
        initial=state_hash(encoder,projector)
        bn_before={k:v.detach().clone() for k,v in encoder.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
        optimizer=torch.optim.AdamW(list(encoder.parameters())+list(projector.parameters()),lr=1e-4,weight_decay=1e-4)
        losses=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output_tokens=study.teaching_tokens(encoder(study.normalize_pixels(pixels[step*16:(step+1)*16],device)),projector)
            loss=study.distillation_loss(output_tokens,torch.from_numpy(targets[step*16:(step+1)*16]).to(device)).mean()
            study.require(torch.isfinite(loss),'Nonfinite engineering loss')
            loss.backward()
            study.require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters()),'Missing/invalid projector gradients')
            study.require(any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.parameters()),'Student encoder is not learning')
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        for key,value in bn_before.items():
            study.require(torch.equal(value,encoder.state_dict()[key]),'BatchNorm running statistics changed')
        final=state_hash(encoder,projector)
        study.require(final!=initial,'Engineering optimizer changed no parameters')
        runs.append({'initialSha256':initial,'finalSha256':final,'losses':losses,'batchNormStatisticsUnchanged':True})
    study.require(runs[0]==runs[1],'Student fitting not deterministic')
    report={'passed':True,'source':identity(__file__),'studentSource':identity(study.__file__),
            'protocol':identity(study.OUTPUT/'protocol.md'),'records':records,'repeatRuns':runs,
            'onePhysicalGpuSharedWithSporadicFp16Encoder':True,
            'noHeldSourceRallyOutcomesUsed':True,'protectedTestOpened':False}
    write_immutable(study.OUTPUT/'distillation-engineering-v1/report.json',report)
    print(json.dumps({'passed':True,'report':str(study.OUTPUT/'distillation-engineering-v1/report.json'),'losses':runs[0]['losses']}))


if __name__=='__main__':
    main()
