import styles from "./button.module.css"

export default function CustomButton({ children, className, loading, ...rest }) {
  return (
    <button className={`${styles.button} ${className} `} disabled={loading} {...rest}>
      {children}
    </button>
  )
}
