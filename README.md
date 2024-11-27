# T-MASS-dataleakage
**(Here is the archive of the issue we raised, in both English and Chinese)**

Hello, Wang,

I’ve been closely following your work on CVPR 2024, `T-MASS`, and after reading your paper at the end of September, I found the approach very interesting and novel. I decided to replicate the paper and test the code, and attempted to improve it. I spent over a month making dozens of modifications. However, a few days ago, while modifying your test code, I found that it was very complex, with dimensions constantly switching back and forth. I decided to improve several functions in your test code. This led to us discovering a serious data leakage issue in your testing section, as well as numerous other inconsistencies.

1. In the `sim_matrix_inference_stochastic` function in your `modules\metrics.py` file, the annotation for `text_embeds_per_video_id.shape` is reversed. After verifying, we found that the first dimension should be `num_vids`, and the second dimension should be `num_txts`. However, you have swapped these dimensions, which represent the video and text modalities. In your `sim_matrix_inference_stochastic_light_allops` function, although it’s not called, its input is the same as the `sim_matrix_inference_stochastic` function, including `text_embeds_per_video_id`, but you give the correct dimensions of video and text in this function. I’m not sure whether this was intentional. Since your test method assumes an equal number of videos and texts, this problem is difficult to detect. However, if someone tried to modify it to search for one text among 1000 videos, the issue would become apparent. This leads to data leakage starting from the batch matrix multiplication `sims = torch.bmm(text_embeds_per_video_id, vid_embeds_pooled_per_video_id)`. At this point, the first dimension of the two matrices corresponds to video and text, and the batch multiplication becomes the beginning of the leakage. Later, when calculating `sims_diag` (the diagonal matrix), the first and last dimensions are not aligned, leading to the introduction of the ground truth(the diagonal) for video-text retrieval test, which causes data leakage.

2. In your test code, after using matrix multiplication in the function `_valid_epoch_step` to calculate `sim_select` (a matrix of shape #trials * bs_t), the result is a similarity matrix of `bs_t` texts with a single video. You can simply select the maximum value from the `#trials` values in `sim_select` for each text to get the best similarity between the video and the `bs_t` texts. After that, you can just stack these results for all videos to obtain the `bs_v * bs_t` similarity matrix. The long testing code afterward is unnecessary and actually introduces data leakage. After modifying the code in this way, the test result was R@1=41.9 (see the figure below), while my single-GPU replication result before modification was 49.4. This further confirms that the significant improvement in your test results is due to data leakage:

   ![8a2e45fae78bc42cb09841d99e84915.png](https://s2.loli.net/2024/11/06/z2bWS8o6fuy1YVe.png)

3. Through our experiments, we found that your `support_loss` is actually a negative optimization. Not to mention that you didn’t even use `support_loss` for datasets other than MSRVTT, we tried removing `support_loss` and only using the other parts of the loss on MSRVTT, and found that the performance actually improved:

   ![image.png](https://s2.loli.net/2024/11/06/mlVTSLjt92Z8YW6.png)

4. The `generate_embeds_per_video_id_stochastic` function in your test code is over 40 lines long, with nested loops, making it overly complex. It also requires the number of videos and texts to be the same, which is not necessary. The code is not elegant and makes it difficult for others to read. Since you didn’t test on MSVD, where the number of videos and texts is not equal, this could be simplified into just two lines, as follows:

   ```python
   text_embeds_per_video_id = text_embeds_stochastic_allpairs.unsqueeze(2)
   vid_embeds_pooled_per_video_id = vid_embeds_pooled.unsqueeze(2)
   ```

   After our validation, these two lines are equivalent to your 40-line-long function.

5. Although `max_text_per_vid` is part of X-Pool, it has no meaning for your work, because you didn’t use MSVD for testing, and in your case, every video has a one-to-one mapping with the text. This means `max_text_per_vid` is always 1, which can confuse the dimensions of video and text: one dimension represents text (`num_txt`), and you’ve turned it into `num_vid * max_text_per_vid`, treating it as the video dimension. This makes the diagonal extraction operation look plausible, even though it’s not.

6. Due to the issues above in the test code, your code can only handle cases where the number of videos and texts are the same, which does not match the real-world scenario of text-to-video retrieval. For example, if I want to provide a text and find its corresponding video from 1000 videos, your code cannot handle this situation. On the other hand, other TVR works like CLIP4Clip have test code that can handle such scenarios.

In conclusion, we believe that your paper’s code suffers from data leakage, which leads to the inflated results. After removing the data leakage, the results were actually worse than the XPool results referenced in your paper, meaning that `performance increase duo to data-leakage + performance decrease duo to negative optimization of your "improvements" = performance increase that looks smaller and thus seems not to have data leakage.` " This is why we initially didn’t suspect data leakage as the cause.

Therefore, we hope you will retract the paper, as its existence not only wasted over a month of our research time but will also waste more time and effort for others, hindering progress in this area.

We look forward to your response.

---

你好，Wang同学，我最近正关注你的CVPR2024工作 `T-MASS` ，从今年9月末读了你的文章发现这篇文章思路很有趣、很新颖，于是把这篇文章复现并且对于代码效果进行复现并且尝试进行改进，前前后后修改了数十个版本，花费了一个多月时间。可是，前几天我在修改你的测试代码的时候，由于你的测试代码写的十分繁琐，各个维度之间前后颠来倒去，我决定对于你的测试代码的几个函数进行改进。由此，我们发现了你的测试部分存在严重的数据泄露问题、也发现了诸多不合理之处。

1. 在你的 `modules\metrics.py` 文件的 `sim_matrix_inference_stochastic` 函数中，`text_embeds_per_video_id.shape` 的标注就搞反了，经过我们的验证，第一个维度应该为 `num_vids` 第二个维度应该为 `num_txts` ，而你正好把这两个维度，也就是代表视频和文本两个模态的维度搞混了。而你写的 `sim_matrix_inference_stochastic_light_allops` 函数，虽然没有被调用，但是二者的输入是一样的，都有 `text_embeds_per_video_id` ，可是这里面你就把视频和文本的维度标注对了，不知道你是不是有意为之，调换了两个维度。因为你的测试方式是视频和文本的数量相同，因此这个问题很难被发现。这也导致了如果有人想要修改成一个一个文本去检索1000个视频，那么他将发现测试的问题所在。  
   因此后面在做 `sims = torch.bmm(text_embeds_per_video_id, vid_embeds_pooled_per_video_id)` 这个批量矩阵乘法的时候，两个矩阵的第一个维度分别为视频和文本，此时的批量乘法就是泄露的开始。而你后面在求 `sims_diag` 也就是取对角阵的时候第一个维度和最后一个维度代表的并不是相同的维度，此时的对角阵相当于引入了视频文本检索对角线的 ground truth ，发生数据泄露。
2. 你的测试代码中，在函数 `_valid_epoch_step` 用矩阵乘法计算出 `sim_select` （#trials * bs_t）后，得到的就是bs_t个文本和单个视频的相似度矩阵，你只需要针对每一个文本选用sim_select中#trials个值中最大的就得到了一个视频与bs_t个文本的相似度，后面你只需要将for循环通过之后将所有视频的这个结果摞在一起即可得到 bs_v * bs_t的相似度矩阵，这后面的大篇幅的测试代码完全没有必要，反而会带来数据泄露。这种修改后得到的测试结果为R@1=41.9（如下图），而修改前我的单卡复现结果为49.4。这也进一步印证了后面测试结果大幅提升就是因为数据泄露：
   ![8a2e45fae78bc42cb09841d99e84915.png](https://s2.loli.net/2024/11/06/z2bWS8o6fuy1YVe.png)
3. 你的 support_loss 经过我们的实验发现完全是负优化，先不说你在除了MSRVTT数据集上根本就没有使用 support_loss ，我们在MSRVTT数据集上尝试将support_loss删除只留下另一部分loss，发现效果反而会有所提升：
   ![image.png](https://s2.loli.net/2024/11/06/mlVTSLjt92Z8YW6.png)
4. 你测试代码中的 `generate_embeds_per_video_id_stochastic` 函数写了四十多行，大循环套小循环，十分复杂，还约束了视频和文本的数量必须一致，这代码写的实在是不优雅，给别人阅读带来了很大困难。因为你没有测MSVD，测试时视频和文本数量一致，其实完全可以简化成两行，如下代码：

```python
            text_embeds_per_video_id = text_embeds_stochastic_allpairs.unsqueeze(2)
            vid_embeds_pooled_per_video_id = vid_embeds_pooled.unsqueeze(2)
```

​	经过我们的验证，完全是等效的...

4. `max_text_per_vid` 虽然是X-Pool中的，但是对于这篇工作，引入没有任何意义，因为你没有使用MSVD用于测试，所有的文本和视频数量都是一对一的，这样 `max_text_per_vid` 都是1，反而会对视频和文本的维度产生混淆：一个维度本身代表文本 `(num_txt)` ，你把它拆成视频*1 `(num_vid * max_text_per_vid)` ，就当成了视频维度，这也使得那个取对角阵的操作看上去很容易被人当成是合理的。

5. 因为以上的测试代码书写问题，导致你的测试代码只能处理视频和文本数量一致的情况，这与实际应用中的文本视频检索场景不符，我想要给出一个文本，在1000个视频里找出它的对应视频，这种你就无法处理，而其他的TVR工作如 CLIP4Clip 的测试代码则是能够处理如此情况的。

综上所述，我们认为你这篇论文的代码出现了数据泄露问题，因而导致效果的提升。而我们去掉数据泄露部分后，效果甚至不如你的论文中参考的XPool的结果，也就是说 `数据泄露导致的大幅提升 + “改进”的负优化造成的效果下降 = 看起来较少的因而不像是数据泄露的效果提升` ，这也是一开始我们并没有怀疑你是数据泄露的原因。

因此我们希望你能够撤稿，因为这篇论文的存在，不仅是对于我们一个多月科研时间的浪费，也会浪费更多人的时间精力，阻碍此方向研究的进展。

希望能得到你的回复！





