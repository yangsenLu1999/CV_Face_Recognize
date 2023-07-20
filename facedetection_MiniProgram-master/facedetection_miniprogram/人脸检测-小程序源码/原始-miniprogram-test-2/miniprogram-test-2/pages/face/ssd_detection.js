//index.js
//获取应用实例
var time = null;
var myCanvas = null;
var windowHeight, windowWidth;
var type = null;
Page({
  data: {
    device:true,
    camera: true,
    x1: 0,
    y1: 0,
    x2: 0,
    y2: 0,
  },
  onLoad() {
    this.setData({
      ctx: wx.createCameraContext(),
      device: this.data.device,
    })
    wx.getSystemInfo({
      success: function (res) {
        console.log(res);
        // 屏幕宽度、高度
        windowHeight = res.windowHeight;
        windowWidth = res.windowWidth;
        console.log('height=' + res.windowHeight);
        console.log('width=' + res.windowWidth);
      }
    })
  },
  open() {
  },
  // 关闭模拟的相机界面
  close() {
  },
})
 