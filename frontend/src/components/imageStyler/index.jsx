import { useContext, useEffect, useState } from "react"
import styles from "./imageStyler.module.css"
import { SelectedImagesContext } from "../../context/selectedImagesContext"
import CustomSlider from "../customSlider"
import { LinkOffOutlined, LinkOutlined, LockOpenOutlined, LockOutlined } from "@mui/icons-material"
import StyleTransferPane from "../styleTransferPane"

function ImageStyler() {
  const { activeImage, setSelectedImages, hoodieBackground } = useContext(SelectedImagesContext)
  const [styleObj, setStyleObj] = useState({
    height: "100px",
    width: "100px",
    borderRadius: "5%",
    fadeRadius: 8,
    fadeIntensity: 8,
    backgroundColor: hoodieBackground,
    backgroundBlendMode: "normal",
    // backgroundPosition: "center",
    backgroundX: 50,
    backgroundY: 50,
  })
  const [lock, setLock] = useState(true)
  const [activeTab, setActiveTab] = useState(1)

  function addImage() {
    setSelectedImages((prev) => [...prev, { image: activeImage, styleObj }])
  }

  useEffect(() => {
    setStyleObj((prev) => {
      return { ...prev, height: prev.width }
    })
  }, [lock])

  return (
    <div className={styles.container}>
      <div style={{ background: hoodieBackground }} className={styles.imageContainer}>
        {activeImage ? (
          <div
            style={{
              width: styleObj.width,
              height: styleObj.height,
              borderRadius: styleObj.borderRadius,
              backgroundImage: `url(${activeImage})`,
              backgroundColor: hoodieBackground,
              backgroundSize: "cover",
              boxShadow: `0 0 ${styleObj.fadeRadius}px ${styleObj.fadeIntensity}px ${hoodieBackground} inset`,
              backgroundBlendMode: styleObj.backgroundBlendMode,
              // backgroundPosition: styleObj.backgroundPosition,
              backgroundPosition: `${styleObj.backgroundX}% ${styleObj.backgroundY}%`,
            }}
          />
        ) : (
          "Please select an image"
        )}
      </div>
      <div className={styles.imageStyleOptions}>
        <div className={styles.sectionTab}>
          <p
            onClick={() => setActiveTab(1)}
            className={styles.sectionHeading}
            style={{ borderBottom: activeTab == 1 ? "2px solid #555" : "none" }}
          >
            Image Styles
          </p>
          <p
            onClick={() => setActiveTab(2)}
            className={styles.sectionHeading}
            style={{ borderBottom: activeTab == 2 ? "2px solid #555" : "none" }}
          >
            Style Transfer
          </p>
        </div>

        {activeTab == 1 && (
          <div style={{ maxHeight: "350px", height: "350px", overflowY: "auto", overflowX: "hidden" }}>
            <p className={styles.inputContainer}>
              <label>Border Radius</label>
              <CustomSlider
                type='range'
                min='0'
                max='100'
                value={styleObj.borderRadius.substring(0, styleObj.borderRadius.length - 1)}
                onChange={(e) => setStyleObj((prev) => ({ ...prev, borderRadius: e.target.value + "%" }))}
                labelValue={styleObj.borderRadius}
              />
            </p>
            <div style={{ position: "relative" }}>
              <p className={styles.inputContainer}>
                <label>Height</label>
                <CustomSlider
                  type='range'
                  min='1'
                  max='200'
                  value={styleObj.height.substring(0, styleObj.height.length - 2)}
                  onChange={(e) =>
                    setStyleObj((prev) => {
                      if (lock) return { ...prev, height: e.target.value + "px", width: e.target.value + "px" }
                      return { ...prev, height: e.target.value + "px" }
                    })
                  }
                  labelValue={styleObj.height}
                />
              </p>
              <div className={styles.lock} onClick={() => setLock(!lock)}>
                <p>{lock ? <LinkOutlined /> : <LinkOffOutlined />}</p>
              </div>
              <p className={styles.inputContainer} style={{ marginTop: 0 }}>
                <label>Width</label>
                <CustomSlider
                  type='range'
                  min='1'
                  max='200'
                  value={styleObj.width.substring(0, styleObj.width.length - 2)}
                  onChange={(e) =>
                    setStyleObj((prev) => {
                      if (lock) return { ...prev, height: e.target.value + "px", width: e.target.value + "px" }
                      return { ...prev, width: e.target.value + "px" }
                    })
                  }
                  labelValue={styleObj.width}
                />
              </p>
            </div>
            <p className={styles.inputContainer}>
              <label>Background X</label>
              <CustomSlider
                type='range'
                min='0'
                max='100'
                value={styleObj.backgroundX}
                onChange={(e) => setStyleObj((prev) => ({ ...prev, backgroundX: e.target.value }))}
                labelValue={styleObj.backgroundX}
              />
            </p>
            <p className={styles.inputContainer}>
              <label>Background Y</label>
              <CustomSlider
                type='range'
                min='0'
                max='100'
                value={styleObj.backgroundY}
                onChange={(e) => setStyleObj((prev) => ({ ...prev, backgroundY: e.target.value }))}
                labelValue={styleObj.backgroundY}
              />
            </p>
            <p className={styles.inputContainer}>
              <label>Fade Radius</label>
              <CustomSlider
                type='range'
                min='0'
                max='100'
                value={styleObj.fadeRadius}
                onChange={(e) => setStyleObj((prev) => ({ ...prev, fadeRadius: e.target.value }))}
                labelValue={styleObj.fadeRadius}
              />
            </p>
            <p className={styles.inputContainer}>
              <label>Fade Intensity</label>
              <CustomSlider
                type='range'
                min='0'
                max='100'
                value={styleObj.fadeIntensity}
                onChange={(e) => setStyleObj((prev) => ({ ...prev, fadeIntensity: e.target.value }))}
                labelValue={styleObj.fadeIntensity}
              />
            </p>
            <p className={styles.inputContainer}>
              <label>Background Blend Mode</label>
              <select
                value={styleObj.backgroundBlendMode}
                onChange={(e) => setStyleObj((prev) => ({ ...prev, backgroundBlendMode: e.target.value }))}
              >
                {[
                  "normal",
                  "multiply",
                  "overlay",
                  "screen",
                  "darken",
                  "lighten",
                  "color-dodge",
                  "color-burn",
                  "hard-light",
                  "soft-light",
                  "difference",
                  "exclusion",
                  "hue",
                  "saturation",
                  "color",
                  "luminosity",
                ].map((val, index) => (
                  <option key={index} value={val}>
                    {val}
                  </option>
                ))}
              </select>
            </p>
          </div>
        )}

        {activeTab == 2 && (
          <div style={{ maxHeight: "350px", height: "350px", overflowY: "auto", overflowX: "hidden" }}>
            <StyleTransferPane image={activeImage} />
          </div>
        )}

        <button className={styles.button} onClick={addImage}>
          Add Image
        </button>
      </div>
    </div>
  )
}

export default ImageStyler

{
  /* <img
          key={styleObj}
          //   onClick={() => onClickHandler(imageIndex, image)}
          style={styleObj}
          // style={image.styleObj}
          //   width={100}
          // className={activeImage?.imageIndex == imageIndex ? styles.activeImage : ""}
          src={activeImage}
        /> */
}
{
  /* <div
        style={{
          width: 100,
          height: 100,
          backgroundImage: `url(${activeImage})`,
          backgroundSize: "cover",
          boxShadow: "0 0 8px 8px white inset",
        }}
      /> */
}
