import styles from "./customSlider.module.css"

function CustomSlider({ value, onChange, min = 0, max = 100, labelValue = null }) {
  return (
    <span className={styles.sliderContainer}>
      <input type='range' min={min} max={max} value={value} onChange={onChange} className={styles.slider} />
      {labelValue && <span>{labelValue}</span>}
    </span>
  )
}

export default CustomSlider
