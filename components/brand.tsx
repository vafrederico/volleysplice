import Image from "next/image";
import Link from "next/link";

type BrandProps = {
  className?: string;
  label: string;
  priority?: boolean;
};

export function Brand({ className, label, priority = false }: BrandProps) {
  return (
    <Link className={className} href="/" aria-label={`VolleySplice ${label}`}>
      <Image
        src="/volleysplice-logo.png"
        alt="VolleySplice"
        width={920}
        height={310}
        priority={priority}
      />
      <span>{label}</span>
    </Link>
  );
}
