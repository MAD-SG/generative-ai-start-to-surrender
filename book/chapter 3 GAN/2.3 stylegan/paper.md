
每一层的风格调制过程
```
def forward(self, x, w):
    style = self.forward_style(w)
    style_split = style.view(-1, 2, self.out_channels, 1, 1)
    x = x * (style_split[:, 0] + 1) + style_split[:, 1]
    return x, style
```
对featuremap的每一层利用学习到的风格向量进行风格调制。具体来说style_split 每个channel 包含一个学习到的标准差和均值
```
x = x * (style_split[:, 0] + 1) + style_split[:, 1]
```
意味最每一个channel 的feature map显式得调节方差和标准差。